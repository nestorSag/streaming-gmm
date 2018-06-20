package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
  
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.mllib.util.{Loader, MLUtils, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}


class SGDGMM(
  private var weights: Array[Double],
  private var gaussians: Array[UpdatableMultivariateGaussian],
  private var optimizer: GMMGradientAscent) extends Serializable {

  def getWeights: Array[Double] = weights
  def getGaussians: Array[UpdatableMultivariateGaussian] = gaussians

  var maxGradientIters = 100
  var convergenceTol = 1e-6
  def k: Int = weights.length

  require(weights.length == gaussians.length, "Length of weight and Gaussian arrays must match")




  def getConvergenceTol: Double = convergenceTol

  def setConvergenceTol(x: Double): this.type = {
    require(x>0,"convergenceTol must be positive")
    convergenceTol = x
    this
  }




  def setmaxGradientIters(maxGradientIters: Int): this.type = {
    require(maxGradientIters > 0 ,s"maxGradientIters needs to be a positive integer; got ${maxGradientIters}")
    this.maxGradientIters = maxGradientIters
    this
  }

  def getmaxGradientIters: Int = {
    this.maxGradientIters
  }




  def setOptimizer(optim: GMMGradientAscent): this.type = {
    this.optimizer = optim
    this
  }

  def getOpimizer: GMMGradientAscent = this.optimizer


  def setLearningRate(alpha: Double): this.type = {
    require(alpha > 0,
      s"learning rate must be positive; got ${alpha}")
    this.optimizer.setLearningRate(alpha)
    this
  }

  def getLearningRate: Double = optimizer.learningRate



  def setWeightLearningRate(alpha: Double): this.type = {
    require(alpha > 0,
      s"weight learning rate must be positive; got ${alpha}")
    this.optimizer.setWeightLearningRate(alpha)
    this
  }

  def getWeightLearningRate: Double = optimizer.weightLearningRate



  // Spark vector predict methods
  def predict(points: RDD[SV]): RDD[Int] = {
    val responsibilityMatrix = predictSoft(points)
    responsibilityMatrix.map(r => r.indexOf(r.max))
  }


  def predict(point: SV): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  def predict(points: JavaRDD[SV]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]


  def predictSoft(points: RDD[SV]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val bcDists = sc.broadcast(gaussians)
    val bcWeights = sc.broadcast(weights)
    points.map { x =>
      computeSoftAssignments(new BDV[Double](x.toArray), bcDists.value, bcWeights.value, k)
    }
  }


  def predictSoft(point: SV): Array[Double] = {
    computeSoftAssignments(new BDV[Double](point.toArray), gaussians, weights, k)
  }

  def predict(point: BDV[Double]): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  def predictSoft(point: BDV[Double]): Array[Double] = {
    computeSoftAssignments(point, gaussians, weights, k)
  }


  private def computeSoftAssignments(
      pt: BDV[Double],
      dists: Array[UpdatableMultivariateGaussian],
      weights: Array[Double],
      k: Int): Array[Double] = {
    val p = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.pdf(pt) //ml eps
    }
    val pSum = p.sum
    for (i <- 0 until k) {
      p(i) /= pSum
    }
    p
  }

  def step(data: RDD[SV]): Unit = {

    val sc = data.sparkContext

    val gconcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))} // y = [x 1]
 
    val d = gconcaveData.first().length - 1

    val shouldDistribute = shouldDistributeGaussians(k, d)

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0
    

    var regVals = Array.fill(k)(0.0)

    val bcOptim = sc.broadcast(this.optimizer)

   // var rv: Array[Double]
     
    while (iter < maxGradientIters && math.abs(newLL-oldLL) > convergenceTol) {

      val compute = sc.broadcast(SampleAggregator.add(weights, gaussians)_)

      val sampleStats = gconcaveData.treeAggregate(SampleAggregator.zero(k, d))(compute.value, _ += _)

      val n: Double = sampleStats.gConcaveCovariance.map{case x => x(d,d)}.sum // number of data points 

      val tuples =
          Seq.tabulate(k)(i => (sampleStats.gConcaveCovariance(i), 
                                this.gaussians(i),
                                this.weights(i)))



      // update gaussians
      var (newRegVal, newDists) = if (shouldDistribute) {
        // compute new gaussian parameters and regularization values in
        // parallel

        val numPartitions = math.min(k, 1024)

        val (rv,gs) = sc.parallelize(tuples, numPartitions).map { case (cov,g,w) =>

          val regVal =  bcOptim.value.penaltyValue(g,w)
          g.update(g.paramMat + bcOptim.value.direction(g,cov) * bcOptim.value.learningRate/n)

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
      weights = getUpdatedWeights(sampleStats,n)


      oldLL = newLL // current becomes previous
      newLL = sampleStats.qLoglikelihood + newRegVal.sum// this is the freshly computed log-likelihood plus regularization


      iter += 1
      compute.destroy()
    }

    //this.gaussians = gaussians
    //this.weights = weights
  }

  def getUpdatedWeights(sampleStats:SampleAggregator, n:Double): Array[Double] = {
    val softmaxWeights = weights.map{case x => math.log(x/weights.last)}

    // weight tuples
    var weightTuples =
        Seq.tabulate(k)(i => (softmaxWeights(i), 
                              this.weights(i),
                              sampleStats.gConcaveCovariance(i),
                              (1 to k)(i)))

    val updatedSMweights = weightTuples.map{ case (smw,w,cov,i) => 
      smw + optimizer.weightLearningRate*optimizer.weightGradient(cov,w,n,i==k)/n
    }

    val totalSoftmax = updatedSMweights.map{case w => math.exp(w)}.sum

    updatedSMweights.map{case w => math.exp(w)/totalSoftmax}.toArray

  }

  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25

}


private[streamingGmm] class SampleAggregator(
  var qLoglikelihood: Double,
  val gConcaveCovariance: Array[BDM[Double]]) extends Serializable{

  val k = gConcaveCovariance.length

  def +=(x: SampleAggregator): SampleAggregator = {
    var i = 0
    while (i < k) {
      gConcaveCovariance(i) += x.gConcaveCovariance(i)
      i += 1
    }
    qLoglikelihood += x.qLoglikelihood
    this
  }

}

private[streamingGmm] object SampleAggregator {

  def zero(k: Int, d: Int): SampleAggregator = {
    new SampleAggregator(0.0,Array.fill(k)(BDM.zeros[Double](d+1, d+1)))
  }

  //compute cluster contributions for each input point
  // (U, T) => U for aggregation
  def add(
      weights: Array[Double],
      dists: Array[UpdatableMultivariateGaussian])
      (agg: SampleAggregator, y: BDV[Double]): SampleAggregator = {

    val q = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.gConcavePdf(y) // <--q-logLikelihood
    }
    val qSum = q.sum
    agg.qLoglikelihood += math.log(qSum)
    var i = 0
    while (i < agg.k) {
      q(i) /= qSum 
      agg.gConcaveCovariance(i) += y*y.t*q(i)
      i = i + 1
    }
    agg
  }
}