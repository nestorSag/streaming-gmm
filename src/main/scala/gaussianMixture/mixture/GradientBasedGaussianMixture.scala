package net.github.gradientgmm

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV, Vectors => SVS, Matrices => SMS}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMG}


import org.apache.log4j.Logger

class GradientBasedGaussianMixture private (
  w:  WeightsWrapper,
  g: Array[UpdatableGConcaveGaussian],
  private[gradientgmm] var optimizer: GMMOptimizer) extends UpdatableGaussianMixture(w,g) with Optimizable {



  var batchFraction = 1.0



  def step(data: RDD[SV]): Unit = {

    val logger: Logger = Logger.getLogger("modelPath")

    val sc = data.sparkContext

    val gConcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))} // y = [x 1]
  
    val d = gConcaveData.first().length - 1

    val shouldDistribute = shouldDistributeGaussians(k, d)

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0
    
    val bcOptim = sc.broadcast(this.optimizer)

    val initialRate = optimizer.learningRate

    // this is to prevent that 0 size samples are too frequent
    // this is because of the way spark takes random samples
    //Prob(0 size sample) <= 1e-3
    val dataSize = gConcaveData.count()
    val minSafeBatchSize: Double = dataSize*(1 - math.exp(math.log(1e-3/dataSize)))

    batchFraction = if(optimizer.batchSize.isDefined){
      math.max(optimizer.batchSize.get.toDouble,minSafeBatchSize)/dataSize
      }else{
        1.0
      }

    while (iter < optimizer.maxIter && math.abs(newLL-oldLL) > optimizer.convergenceTol) {

      //send values formatted for R processing to logs
      logger.debug(s"means: list(${gaussians.map{case g => "c(" + g.getMu.toArray.mkString(",") + ")"}.mkString(",")})")
      logger.debug(s"weights: ${"c(" + weights.weights.mkString(",") + ")"}")

      val adder = sc.broadcast(
        StatAggregator.add(weights.weights, gaussians, optimizer, dataSize*batchFraction)_)

      //val x = batch(gConcaveData)
      //logger.debug(s"sample size: ${x.count()}")
      val sampleStats = batch(gConcaveData).treeAggregate(StatAggregator.init(k, d))(adder.value, _ += _)

      val n: Double = sampleStats.posteriors.sum // number of data points 
      logger.debug(s"n: ${n}")

      val tuples =
          Seq.tabulate(k)(i => (sampleStats.gradients(i), 
                                gaussians(i)))



      // update gaussians
      var newDists = if (shouldDistribute) {
        // compute new gaussian parameters and regularization values in
        // parallel

        val numPartitions = math.min(k, 1024)

        val newDists = sc.parallelize(tuples, numPartitions).map { case (grad,dist) =>

          dist.update(dist.paramMat + grad * bcOptim.value.learningRate)

          bcOptim.value.updateLearningRate //update learning rate in workers

          dist

        }.collect()

        newDists.toArray

      } else {

        val newDists = tuples.map{ 
          case (grad,dist) => 

          dist.update(dist.paramMat + grad * optimizer.learningRate)
          
          dist

        }

        newDists.toArray

      }

      gaussians = newDists

      val posteriorResps = sampleStats.posteriors

      //update weights in the driver
      val current = optimizer.fromSimplex(Utils.toBDV(weights.weights))
      val delta = optimizer.learningRate/n*optimizer.softWeightsDirection(Utils.toBDV(posteriorResps),weights)
      weights.update(optimizer.toSimplex(current + delta))

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

  def toSparkGMM: GaussianMixtureModel = {

    val d = gaussians(0).getMu.length

    new GaussianMixtureModel(
      weights.weights,
      gaussians.map{
        case g => new SMG(
          SVS.dense(g.getMu.toArray),
          SMS.dense(d,d,g.getSigma.toArray))})
  }



  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25



  private def batch(data: RDD[BDV[Double]]): RDD[BDV[Double]] = {
    if(batchFraction < 1.0){
      data.sample(false,batchFraction)
    }else{
      data
    }
  }

}

object GradientBasedGaussianMixture{

  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGConcaveGaussian],
    optimizer: GMMOptimizer): GradientBasedGaussianMixture = {
    new GradientBasedGaussianMixture(new WeightsWrapper(weights),gaussians,optimizer)
  }

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
      new WeightsWrapper(proportions.map{case p => p/(n+k)}), 
      (0 to k-1).map{case i => UpdatableGConcaveGaussian(means(i),pseudoCov(i))}.toArray,
      optimizer)

  }

  private def initCovariance(x: IndexedSeq[BDV[Double]]): BDM[Double] = {
    val mu = vectorMean(x)
    val ss = BDV.zeros[Double](x(0).length)
    x.foreach(xi => ss += (xi - mu) ^:^ 2.0)
    diag(ss / x.length.toDouble)
  }

  private def vectorMean(x: IndexedSeq[BV[Double]]): BDV[Double] = {
    val v = BDV.zeros[Double](x(0).length)
    x.foreach(xi => v += xi)
    v / x.length.toDouble
  }
}