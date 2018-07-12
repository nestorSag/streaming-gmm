package com.github.nestorsag.gradientgmm

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV, Vectors => SVS, Matrices => SMS}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMG}


import org.apache.log4j.Logger

/**
  * Optimizable gradient-based Gaussian Mixture Model
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param w Weight vector wrapper
  * @param g Array of mixture components (distributions)
  * @param optimizer Optimization object
 
  */
class GradientBasedGaussianMixture private (
  w:  UpdatableWeights,
  g: Array[UpdatableGaussianMixtureComponent],
  private[gradientgmm] var optimizer: GMMOptimizer) extends UpdatableGaussianMixture(w,g) with Optimizable {


/**
  * mini-batch size as fraction of the complete training data
 
  */
  var batchFraction = 1.0


/**
  * Optimize the mixture parameters given some training data
  * @param data Training data as an RDD of Spark vectors 
 
  */
  def step(data: RDD[SV]): Unit = {

    // initialize logger. It logs the parameters' paths to solution
    // the messages' leve; lis set to DEBUG, so be sure to set the log level to DEBUG 
    // if you want to see them
    val logger: Logger = Logger.getLogger("modelPath")

    val sc = data.sparkContext

    //map original vectors to points for the g-concave formulation
    val gConcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))} // y = [x 1]
    
    val d = gConcaveData.first().length - 1

    val shouldDistribute = shouldDistributeGaussians(k, d)

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0
    
    // breadcast optimizer to workers
    val bcOptim = sc.broadcast(this.optimizer)

    val initialRate = optimizer.learningRate

    // this is to prevent that 0 size samples are too frequent
    // this is because of the way spark takes random samples
    //Prob(0 size sample) <= 1e-3
    val dataSize = gConcaveData.count()
    val minSafeBatchSize: Double = dataSize*(1 - math.exp(math.log(1e-3)/dataSize))

    batchFraction = if(optimizer.batchSize.isDefined){
      math.max(optimizer.batchSize.get.toDouble,minSafeBatchSize)/dataSize
      }else{
        1.0
      }

    while (iter < optimizer.maxIter && math.abs(newLL-oldLL) > optimizer.convergenceTol) {

      //send values formatted for R processing to logs
      logger.debug(s"means: list(${gaussians.map{case g => "c(" + g.getMu.toArray.mkString(",") + ")"}.mkString(",")})")
      logger.debug(s"weights: ${"c(" + weights.weights.mkString(",") + ")"}")

      // initialize curried adder that will aggregate the necessary statistics in the workers
      // dataSize*batchFraction is the expected current batch size
      // but it is not exact due to how spark takes samples from RDDs
      val adder = sc.broadcast(
        GradientAggregator.add(weights.weights, gaussians, optimizer, dataSize*batchFraction)_)

      //val x = batch(gConcaveData)
      //logger.debug(s"sample size: ${x.count()}")
      val sampleStats = batch(gConcaveData).treeAggregate(GradientAggregator.init(k, d))(adder.value, _ += _)

      val n: Double = sum(sampleStats.weightsGradient) // number of actual data points in current batch
      logger.debug(s"n: ${n}")

      // pair Gaussian components with their respective parameter gradients
      val tuples =
          Seq.tabulate(k)(i => (sampleStats.gaussianGradients(i), 
                                gaussians(i)))



      // update gaussians
      var newDists = if (shouldDistribute) {
        // compute new gaussian parameters and regularization values in
        // parallel

        val numPartitions = math.min(k, 1024)

        val newDists = sc.parallelize(tuples, numPartitions).map { case (grad,dist) =>

          dist.update(
            bcOptim.value.getGaussianUpdate(
              dist.paramMat,
              grad,
              dist.optimUtils))

          //dist.update(dist.paramMat + bcOptim.value.direction(grad,dist.optimUtils) * bcOptim.value.learningRate)

          bcOptim.value.updateLearningRate //update learning rate in workers

          dist

        }.collect()

        newDists.toArray

      } else {

        val newDists = tuples.map{ 
          case (grad,dist) => 

          dist.update(
            optimizer.getGaussianUpdate(
              dist.paramMat,
              grad,
              dist.optimUtils))

          //dist.update(dist.paramMat + grad * optimizer.learningRate)
          
          dist

        }

        newDists.toArray

      }

      gaussians = newDists

      weights.update(
          optimizer.getWeightsUpdate(
            Utils.toBDV(weights.weights),
            sampleStats.weightsGradient,
            weights.optimUtils))


      oldLL = newLL // current becomes previous
      newLL = sampleStats.qLoglikelihood
      logger.debug(s"newLL: ${newLL}")

      optimizer.updateLearningRate //update learning rate in driver
      iter += 1
      adder.destroy()
    }

    bcOptim.destroy()
    optimizer.learningRate = initialRate

  }

/**
  * Returns a Spark's Gaussian Mixture Model with the current parameters initialized with the current parameters
 
  */
  def toSparkGMM: GaussianMixtureModel = {

    val d = gaussians(0).getMu.length

    new GaussianMixtureModel(
      weights.weights,
      gaussians.map{
        case g => new SMG(
          SVS.dense(g.getMu.toArray),
          SMS.dense(d,d,g.getSigma.toArray))})
  }


/**
  * Heuristic to decide when to distribute the computations. Taken from Spark's GaussianMixture class
 
  */
  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25


/**
  * take sample for the current mini-batch, or pass the whole dataset if {{{optimizer.batchSize = None}}}
 
  */
  private def batch(data: RDD[BDV[Double]]): RDD[BDV[Double]] = {
    if(batchFraction < 1.0){
      data.sample(false,batchFraction)
    }else{
      data
    }
  }

}

object GradientBasedGaussianMixture{

/**
  * Creates a new {{{GradientBasedGaussianMixture}}} instance
  * @param weights Array of weights
  * @param gaussians Array of mixture components
  * @param optimizer Optimizer object
 
  */
  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGaussianMixtureComponent],
    optimizer: GMMOptimizer): GradientBasedGaussianMixture = {
    new GradientBasedGaussianMixture(new UpdatableWeights(weights),gaussians,optimizer)
  }

/**
  * Creates a new {{{GradientBasedGaussianMixture}}} instance initialized with the
  * results of a K-means model fitted with a sample of the data
  * @param data training data in the form of an RDD of Spark vectors
  * @param optimizer Optimizer object
  * @param k Number of components in the mixture
  * @param nSamples Number of data points to train the K-means model
  * @param nIters Number of iterations allowed for the K-means model
  * @param seed random seed
  */
  def initialize(
    data: RDD[SV],
    optimizer: GMMOptimizer,
    k: Int,
    nSamples: Int,
    nIters: Int,
    seed: Long = 0): GradientBasedGaussianMixture = {
    
    val sc = data.sparkContext
    val d = data.take(1)(0).size
    val n = math.max(nSamples,2*k)
    var samples = sc.parallelize(data.takeSample(withReplacement = false, n, seed))

    //create kmeans model
    val kmeansModel = new KMeans()
      .setMaxIterations(nIters)
      .setK(k)
      .setSeed(seed)
      .run(samples)
    
    val means = kmeansModel.clusterCenters.map{case v => Utils.toBDV(v.toArray)}

    //add means to sample points to avoid having cluster with zero points 
    samples = samples.union(sc.parallelize(means.map{case v => SVS.dense(v.toArray)}))

    // broadcast values to compute sample covariance matrices
    val kmm = sc.broadcast(kmeansModel)
    val scMeans = sc.broadcast(means)

    // get empirical cluster proportions to initialize the mixture/s weights
    //add 1 to counts to avoid division by zero
    val proportions = samples
      .map{case s => (kmm.value.predict(s),1)}
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()
      .map{case (k,p) => p.toDouble}

    val scProportions = sc.broadcast(proportions)

    //get empirical covariance matrices
    //also add a rescaled identity matrix to avoid starting with singular matrices 
    val pseudoCov = samples
      .map{case v => {
          val prediction = kmm.value.predict(v)
          val denom = math.sqrt(scProportions.value(prediction))
          (prediction,(Utils.toBDV(v.toArray)-scMeans.value(prediction))/denom) }} // x => (x-mean)
      .map{case (k,v) => (k,v*v.t)}
      .reduceByKey(_ + _)
      .map{case (k,v) => {
        val avgVariance = math.max(1e-4,trace(v))/d
        (k,v + BDM.eye[Double](d) * avgVariance)
        }}
      .sortByKey()
      .collect()
      .map{case (k,m) => m}

    new GradientBasedGaussianMixture(
      new UpdatableWeights(proportions.map{case p => p/(n+k)}), 
      (0 to k-1).map{case i => UpdatableGaussianMixtureComponent(means(i),pseudoCov(i))}.toArray,
      optimizer)

  }
}