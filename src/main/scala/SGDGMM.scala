package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.stats.distributions.RandBasis

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.mllib.util.{Loader, MLUtils, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

class SGDGMM private(
  private var weights: Array[Double],
  private var gaussians: Array[UpdatableMultivariateGaussian],
  private var optimizer: GMMGradientAscent) extends Serializable {

  def getWeights: Array[Double] = weights
  def getGaussians: Array[UpdatableMultivariateGaussian] = gaussians

  var maxGradientIters = 100
  var convergenceTol = 1e-6

  var learningRateShrinkage = 1.0
  var minLearningRate = 1e-2

  def k: Int = weights.length

  require(weights.length == gaussians.length, "Length of weight and Gaussian arrays must match")

  def getLearningShrinkage: Double = learningRateShrinkage

  def setLearningShrinkage(x: Double): this.type = {
    require(x>0 && x<=1,"shrinkage must be in [0,1]")
    learningRateShrinkage = x
    this
  }

  def getMinLearningRate: Double = minLearningRate

  def setMinLearningRate(x: Double): this.type = {
    require(x>0 && x < optimizer.learningRate,"minLearningRate must be in [0,learningRate]")
    minLearningRate = x
    this
  }



  def getConvergenceTol: Double = convergenceTol

  def setConvergenceTol(x: Double): this.type = {
    require(x>0,"convergenceTol must be positive")
    convergenceTol = x
    this
  }




  def setmaxGradientIters(x: Int): this.type = {
    require(x > 0 ,s"maxGradientIters needs to be a positive integer; got ${x}")
    maxGradientIters = x
    this
  }

  def getmaxGradientIters: Int = {
    maxGradientIters
  }




  def setOptimizer(optim: GMMGradientAscent): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: GMMGradientAscent = this.optimizer


  def setLearningRate(alpha: Double): this.type = {
    require(alpha > 0,
      s"learning rate must be positive; got ${alpha}")
    optimizer.learningRate = alpha
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





  // Multiple point prediction SV
  def predict(points: RDD[SV]): RDD[Int] = {
    val responsibilityMatrix = predictSoft(points)
    responsibilityMatrix.map(r => r.indexOf(r.max))
  }

  // Multiple point prediction BDV
  // def predict(points: RDD[BDV[Double]]): RDD[Int] = {
  //   val responsibilityMatrix = predictSoft(points)
  //   responsibilityMatrix.map(r => r.indexOf(r.max))
  // }






  // single point prediction BDV
  // def predict(point: BDV[Double]): Int = {
  //   val r = predictSoft(point)
  //   r.indexOf(r.max)
  // }

  // single point prediction SV
  def predict(point: SV): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  // single point prediction JavaRDD
  def predict(points: JavaRDD[SV]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]





  // multiple point predictSoft SV
  def predictSoft(points: RDD[SV]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val bcDists = sc.broadcast(gaussians)
    val bcWeights = sc.broadcast(weights)
    points.map { x =>
      computeSoftAssignments(new BDV[Double](x.toArray), bcDists.value, bcWeights.value, k)
    }
  }


  // multiple point predictSoft SV
  // def predictSoft(points: RDD[BDV[Double]]): RDD[Array[Double]] = {
  //   val sc = points.sparkContext
  //   val bcDists = sc.broadcast(gaussians)
  //   val bcWeights = sc.broadcast(weights)
  //   points.map { case x =>
  //     computeSoftAssignments(x, bcDists.value, bcWeights.value, k)
  //   }
  // }




  // single point predictSoft SV
  def predictSoft(point: SV): Array[Double] = {
    computeSoftAssignments(new BDV[Double](point.toArray), gaussians, weights, k)
  }


  // // single point predictSoft bdv
  // def predictSoft(point: BDV[Double]): Array[Double] = {
  //   computeSoftAssignments(point, gaussians, weights, k)
  // }


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

    val initialLearningRate = optimizer.learningRate

    val sc = data.sparkContext

    val gconcaveData = if(maxGradientIters>1){
      //if iters>1 cache data
      data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))}.cache() // y = [x 1]
    }else{
      data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))} // y = [x 1]
    }
 
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

          bcOptim.value.learningRate *= learningRateShrinkage

          (regVal,g)

        }.collect().unzip

        (rv.toArray,gs.toArray)
      } else {

        val (rv,gs) = tuples.map{ 
          case (cov,g,w) => 

          val regVal = optimizer.penaltyValue(g,w)
          g.update(g.paramMat + optimizer.direction(g,cov) * optimizer.learningRate/n)

          optimizer.learningRate *= learningRateShrinkage

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

    optimizer.learningRate = initialLearningRate

    //this.gaussians = gaussians
    //this.weights = weights
  }

  private def getUpdatedWeights(sampleStats:SampleAggregator, n:Double): Array[Double] = {
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

object SGDGMM{

  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableMultivariateGaussian],
    optimizer: GMMGradientAscent): SGDGMM = {
    new SGDGMM(weights,gaussians,optimizer)
  }

  def apply(
    k: Int,
    d: Int,
    optimizer: GMMGradientAscent): SGDGMM = {

    new SGDGMM(
      (1 to k).map{x => 1.0/k}.toArray,
      (1 to k).map{x => UpdatableMultivariateGaussian(BDV.rand(d),BDM.eye[Double](d))}.toArray,
      optimizer)

  }

  def apply(
    k: Int,
    optimizer: GMMGradientAscent,
    data: RDD[SV],
    seed: Long = 0): SGDGMM = {
    
    val nSamples = 5
    val samples = data.map{x => new BDV[Double](x.toArray)}.takeSample(withReplacement = true, k * nSamples, seed)

    new SGDGMM(
      Array.fill(k)(1.0 / k), 
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