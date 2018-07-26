package com.github.gradientgmm.models

import com.github.gradientgmm.components.{UpdatableGaussianComponent, UpdatableWeights, Utils}
import com.github.gradientgmm.optim.algorithms.{Optimizable, Optimizer, GradientAscent}

import breeze.linalg.{diag, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV, Vectors => SVS, Matrices => SMS}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMG}


import org.apache.log4j.Logger

/**
  * Optimizable gradient-based Gaussian Mixture Model
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param w Weight vector wrapper
  * @param g Array of mixture components (distributions)
  * @param optim Optimization object
 
  */
class GradientGaussianMixture private[models] (
  w:  UpdatableWeights,
  g: Array[UpdatableGaussianComponent],
  var optim: Optimizer) extends UpdatableGaussianMixture(w,g) with Optimizable {


/**
  * mini-batch size as fraction of the complete training data
  * Spark needs a double in [0,1] to take samples, so we need
  * to translate a given batch size to a fraction of the data size
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

    val d = data.first().size

    val sc = data.sparkContext

    //map original vectors to points for the g-concave formulation
    // y = [x 1]
    val gConcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))}.cache()

    val shouldDistribute = shouldDistributeGaussians(k, d)

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0
    
    // broadcast optim to workers
    val bcOptim = sc.broadcast(this.optim)

    val initialRate = optim.getLearningRate



    val(ebs,fraction): (Double,Double) = { //ebs = epected batch size
      val dataSize = gConcaveData.count()

      if(batchSize.isDefined){
        // this is to prevent that 0 size samples are too frequent
        // this is because of the way spark takes random samples
        // we want Prob(0 size sample) <= 1e-3
        val safeBatchSize: Double = dataSize*(1 - math.exp(math.log(1e-3)/dataSize))
        val correctedBatchSize = math.max(batchSize.get.toDouble,safeBatchSize)
        (correctedBatchSize,correctedBatchSize/dataSize)
      }else{
        (dataSize,1.0)
      }
    }
    logger.debug(s"expected batch size: ${ebs}, fraction: ${fraction}")
    batchFraction = math.min(fraction,1.0) //in case batches are larger than whole dataset
    optim.setN(ebs)
    
    //a bit of syntactic sugar
    def toSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.toSimplex
    def fromSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.fromSimplex


    while (iter < maxIter && math.abs(newLL-oldLL) > convergenceTol) {
      val t0 = System.nanoTime

      // if model parameters can be plotted (specific d and k)
      // and logger is set to debug, send parameters' trajectory to logs
      if(d==2 && k == 3){
        //send values formatted for R processing to logs
        logger.debug(s"means: list(${gaussians.map{case g => "c(" + g.getMu.toArray.mkString(",") + ")"}.mkString(",")})")
        logger.debug(s"weights: ${"c(" + weights.weights.mkString(",") + ")"}")
        logger.debug(s"covs: list(${gaussians.map{case g => "c(" + g.getSigma.toArray.mkString(",") + ")"}.mkString(",")})")
      }

      // initialize curried adder that will aggregate the necessary statistics in the workers
      // dataSize*batchFraction is the expected current batch size
      // but it is not exact due to how spark takes samples from RDDs
      val adder = sc.broadcast(
        GradientAggregator.add(weights.weights, gaussians, optim)_)

      //val x = batch(gConcaveData)
      //logger.debug(s"sample size: ${x.count()}")
      val sampleStats = batch(gConcaveData).treeAggregate(GradientAggregator.init(k, d))(adder.value, _ += _)

      val n: Int = sampleStats.counter // number of actual data points in current batch

      if(n>0){
      // pair Gaussian components with their respective parameter gradients
        val tuples =
            Seq.tabulate(k)(i => (sampleStats.gaussianGradients(i) / n.toDouble, //average gradients 
                                  gaussians(i)))

        // update gaussians
        var newDists = if (shouldDistribute) {
          // compute new gaussian parameters and regularization values in
          // parallel

          val numPartitions = math.min(k, 1024)

          val newDists = sc.parallelize(tuples, numPartitions).map { case (grad,dist) =>

            val newPars = bcOptim.value.getUpdate(
                dist.paramMat,
                grad,
                dist.optimUtils)
            
            dist.update(newPars)

            //dist.update(dist.paramMat + bcOptim.value.direction(grad,dist.optimUtils) * bcOptim.value.learningRate)

            bcOptim.value.updateLearningRate //update learning rate in workers

            dist

          }.collect()

          newDists.toArray

        } else {

          val newDists = tuples.map{ 
            case (grad,dist) => 

            dist.update(
              optim.getUpdate(
                dist.paramMat,
                grad,
                dist.optimUtils))
            
            dist

          }

          newDists.toArray

        }

        gaussians = newDists

        //logger.debug(s"weight gradient: ${sampleStats.weightsGradient / n.toDouble}")

        val newWeights = optim.getUpdate(
              fromSimplex(Utils.toBDV(weights.weights)),
              sampleStats.weightsGradient / n.toDouble, //average gradients
              weights.optimUtils)

        weights.update(toSimplex(newWeights))


        oldLL = newLL // current becomes previous
        newLL = sampleStats.loss / ebs //average loss
        logger.trace(s"newLL: ${newLL}")

        optim.updateLearningRate //update learning rate in driver
        iter += 1
        }else{
          logger.debug("No points in sample. Skipping iteration")
        }

      adder.destroy()
      val elapsed = (System.nanoTime - t0)/1e9d
      logger.info(s"iteration ${iter} took ${elapsed} seconds for ${n} samples")
    }

    bcOptim.destroy()

    //set learning rate to original value in case it was shrunk
    optim.setLearningRate(initialRate)

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
  * Update model parameters using streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
  def step(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      step(rdd)
    }
  }

/**
  * Cluster membership prediction for streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
  def predict(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predict(rdd)
    }
  }

/**
  * Soft cluster membership prediction for streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
  def predictSoft(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predictSoft(rdd)
    }
  }


/**
  * Heuristic to decide when to distribute the computations. Taken from Spark's GaussianMixture class
 
  */
  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25


/**
  * take sample for the current mini-batch, or pass the whole dataset if optim.batchSize = None
 
  */
  private def batch(data: RDD[BDV[Double]]): RDD[BDV[Double]] = {
    if(batchFraction < 1.0){
      data.sample(false,batchFraction)
    }else{
      data
    }
  }

}

object GradientGaussianMixture{
/**
  * Creates a new GradientGaussianMixture instance
  * @param weights Array of weights
  * @param gaussians Array of mixture components
  * @param optim Optimizer object
 
  */
  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGaussianComponent]): GradientGaussianMixture = {

    new GradientGaussianMixture(
      new UpdatableWeights(weights),
      gaussians,
      new GradientAscent())
  }

/**
  * Creates a new GradientGaussianMixture instance
  * @param weights Array of weights
  * @param gaussians Array of mixture components
  * @param optim Optimizer object
 
  */
  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGaussianComponent],
    optim: Optimizer): GradientGaussianMixture = {

    new GradientGaussianMixture(
      new UpdatableWeights(weights),
      gaussians,
      optim)
  }

/**
  * Creates a new GradientGaussianMixture instance initialized with the
  * results of a K-means model fitted with a sample of the data
  * @param data training data in the form of an RDD of Spark vectors
  * @param optim Optimizer object
  * @param k Number of components in the mixture
  * @param pointsPerCl The K-Means model will be trained with k*pointsPerCl points
  * @param nIters Number of iterations allowed for the K-means model
  * @param seed random seed
  */
  def initialize(
    data: RDD[SV],
    optim: Optimizer,
    k: Int,
    pointsPerCl: Int = 50,
    nIters: Int = 20,
    seed: Long = 0): GradientGaussianMixture = {
    
    val dataSize = data.count()

    val sc = data.sparkContext
    val d = data.take(1)(0).size //get data dimensionality
    val n = math.min(dataSize,pointsPerCl*k).toInt //in case the data has too few points
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

    new GradientGaussianMixture(
      new UpdatableWeights(proportions.map{case p => p/(n+k)}), 
      (0 to k-1).map{case i => UpdatableGaussianComponent(means(i),pseudoCov(i))}.toArray,
      optim)

  }

  /**
  * Fit a Gaussian Mixture Model (see [[https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model]]).
  * The model is initialized using a K-means algorithm over a small sample and then 
  * fitting the resulting parameters to the data using this {GMMOptimization} object
  * @param data Data to fit the model
  * @param optim Optimization algorithm
  * @param k Number of mixture components (clusters)
  * @param batchSize number of samples processed per iteration
  * @param maxIter maximum number of gradient ascent steps allowed
  * @param convTol log-likelihood change tolerance for stopping criteria
  * @param pointsPerCl The K-Means model will be trained with k*pointsPerCl points
  * @param kMeansIters Number of iterations allowed for the K-means algorithm
  * @param seed Random seed
  * @return Fitted model
  */
  def fit(
    data: RDD[SV], 
    optim: Optimizer = new GradientAscent(), 
    k: Int = 2,
    batchSize: Option[Int] = None,
    maxIter: Int = 100,
    convTol: Double = 1e-6, 
    pointsPerCl: Int = 50,
    kMeansIters: Int = 20, 
    seed: Int = 0): GradientGaussianMixture = {
    
    val model = initialize(
                  data,
                  optim,
                  k,
                  pointsPerCl,
                  kMeansIters,
                  seed)
    
    if(batchSize.isDefined){
      model.setBatchSize(batchSize.get)
    }

    model
    .setMaxIter(maxIter)
    .setConvergenceTol(convTol)
    .step(data)

    model
  }
}