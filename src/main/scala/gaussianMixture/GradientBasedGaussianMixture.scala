package net.github.gradientgmm

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD

import org.apache.log4j.Logger

class GradientBasedGaussianMixture(
  w:  WeightsWrapper,
  g: Array[UpdatableMultivariateGaussian],
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
    

    var regVals = Array.fill(k)(0.0)

    val bcOptim = sc.broadcast(this.optimizer)

    val initialRate = optimizer.learningRate
   // var rv: Array[Double]
    batchFraction = if(batchSize.isDefined){
      batchSize.get.toDouble/gConcaveData.count()
      }else{
        1.0
      }

    while (iter < maxGradientIters && math.abs(newLL-oldLL) > convergenceTol) {

      //send values formatted for R processing to logs
      logger.debug(s"means: list(${gaussians.map{case g => "c(" + g.getMu.toArray.mkString(",") + ")"}.mkString(",")})")
      logger.debug(s"weights: ${"c(" + weights.weights.mkString(",") + ")"}")

      val compute = sc.broadcast(SampleAggregator.add(weights.weights, gaussians)_)

      //val x = batch(gConcaveData)
      //logger.debug(s"sample size: ${x.count()}")
      val sampleStats = batch(gConcaveData).treeAggregate(SampleAggregator.zero(k, d))(compute.value, _ += _)

      val n: Double = sampleStats.gConcaveCovariance.map{case x => x(d,d)}.sum // number of data points 
      logger.debug(s"n: ${n}")

      val tuples =
          Seq.tabulate(k)(i => (sampleStats.gConcaveCovariance(i), 
                                gaussians(i),
                                weights.weights(i)))



      // update gaussians
      var (newRegVal, newDists) = if (shouldDistribute) {
        // compute new gaussian parameters and regularization values in
        // parallel

        val numPartitions = math.min(k, 1024)

        val (rv,gs) = sc.parallelize(tuples, numPartitions).map { case (cov,g,w) =>

          val regVal =  bcOptim.value.penaltyValue(g,w)
          g.update(g.paramMat + bcOptim.value.direction(g,cov) * bcOptim.value.learningRate/n)

          bcOptim.value.updateLearningRate

          (regVal,g)

        }.collect().unzip

        (rv.toArray,gs.toArray)
      } else {

        val (rv,gs) = tuples.map{ 
          case (cov,g,w) => 

          val regVal = optimizer.penaltyValue(g,w)
          g.update(g.paramMat + optimizer.direction(g,cov) * optimizer.learningRate/n)
          
          (regVal, g)

        }.unzip

        (rv.toArray,gs.toArray)

      }

      gaussians = newDists

      val posteriorResps = sampleStats.gConcaveCovariance.map{case x => x(d,d)}

      //update weights in the driver
      val current = optimizer.fromSimplex(Utils.toBDV(weights.weights))
      val delta = optimizer.learningRate/n*optimizer.softWeightsDirection(Utils.toBDV(posteriorResps),weights)
      weights.update(optimizer.toSimplex(current + delta))

      oldLL = newLL // current becomes previous
      newLL = (sampleStats.qLoglikelihood + newRegVal.sum)/n// this is the freshly computed log-likelihood plus regularization
      logger.debug(s"newLL: ${newLL}")

      optimizer.updateLearningRate
      iter += 1
      compute.destroy()
    }

    bcOptim.destroy()
    optimizer.learningRate = initialRate

  }

  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25

  private def batch(data: RDD[BDV[Double]]): RDD[BDV[Double]] = {
    if(batchFraction < 1.0){
      data.sample(true,batchFraction)
    }else{
      data
    }
  }

}

object GradientBasedGaussianMixture{

  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableMultivariateGaussian],
    optimizer: GMMOptimizer): GradientBasedGaussianMixture = {
    new GradientBasedGaussianMixture(new WeightsWrapper(weights),gaussians,optimizer)
  }

  def apply(
    k: Int,
    d: Int,
    optimizer: GMMOptimizer): GradientBasedGaussianMixture = {

    new GradientBasedGaussianMixture(
      new WeightsWrapper((1 to k).map{x => 1.0/k}.toArray),
      (1 to k).map{x => UpdatableMultivariateGaussian(BDV.rand(d),BDM.eye[Double](d))}.toArray,
      optimizer)

  }

  def apply(
    k: Int,
    optimizer: GMMOptimizer,
    data: RDD[SV],
    seed: Long = 0): GradientBasedGaussianMixture = {
    
    val nSamples = 5
    val samples = data.map{x => new BDV[Double](x.toArray)}.takeSample(withReplacement = true, k * nSamples, seed)

    new GradientBasedGaussianMixture(
      new WeightsWrapper(Array.fill(k)(1.0 / k)), 
      Array.tabulate(k) { i =>
      val slice = samples.view(i * nSamples, (i + 1) * nSamples)
      UpdatableMultivariateGaussian(vectorMean(slice), initCovariance(slice))
      },
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

class WeightsWrapper(var weights: Array[Double]) extends Serializable{

  var momentum: Option[BDV[Double]] = None
  var adamInfo: Option[BDV[Double]] = None
  var length = weights.length

  def update(newWeights: BDV[Double]): Unit = {
    // recenter soft weights to avoid under or overflow
    weights = newWeights.toArray
  }

  private[gradientgmm] def updateMomentum(x: BDV[Double]): Unit = {
    momentum = Option(x)
  }

  private[gradientgmm] def updateAdamInfo(x: BDV[Double]): Unit = {
    adamInfo = Option(x)
  }

  private[gradientgmm] def initializeMomentum: Unit = {
    momentum = Option(BDV.zeros[Double](weights.length))
  }

  private[gradientgmm] def initializeAdamInfo: Unit = {
     adamInfo = Option(BDV.zeros[Double](weights.length))
  }

}