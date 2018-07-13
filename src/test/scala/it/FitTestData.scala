import org.scalatest.FlatSpec


import com.github.nestorsag.gradientgmm._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

//this test runs the program over the test data and generate logs that can be parsed
// with the R utils to generate plots of the opimisation path
//make sure to set the log level to DEBUG
object FitTestData{

  var sc : SparkContext = _

  val conf = new SparkConf().setAppName("GradientBasedGaussianMixture-test").setMaster("local[4]")

  val errorTol = 1e-8
  val k = 3

  def paramParser: (Double,Double,Double,Option[Int]) = {
    val source = scala.io.Source.fromFile("src/test/resources/testFitParams.txt")
    val lines = source.getLines.toArray
    source.close()

    var pattern = """(lr=)(.+)""".r
    var pattern(_,lr) = lines(0)

    pattern = """(shrinkageRate=)(.+)""".r
    var pattern(_,shrinkageRate) = lines(1)

    pattern = """(minLr=)(.+)""".r
    var pattern(_,minLr) = lines(2)

    var batchSize:Option[Int] = None

    if(lines.length > 3){
      pattern = """(batchSize=)(.+)""".r
      var pattern(_,batchSizeValue) = lines(3)
      batchSize = Option(batchSizeValue.toInt)

    }

    (lr.toDouble,shrinkageRate.toDouble,minLr.toDouble,batchSize)
  }

  def run: Unit = {

    val (lr,shrinkageRate,minLr,batchSize) = paramParser

    sc = new SparkContext(conf)

    val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
    val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()
    val d = parsedData.take(1)(0).size
      
    val x = parsedData.take(1)(0)

    val initialWeights = (1 to k).map{case x => 1.0/k}.toArray

    val means = Array(BDV(-1.0,0.0),BDV(0.0,-1.0),BDV(1.0,0.0))
    //val means = (1 to k).map{case i => BDV.rand(2)}.toArray
    val covs = (1 to k).map{case k => BDM.eye[Double](d)}.toArray
    val initialDists = means.zip(covs).map{case (m,s) => UpdatableGaussianMixtureComponent(m,s)}

    val optim = new GradientAscent()
     .setLearningRate(lr)
     .setShrinkageRate(shrinkageRate)
     .setMinLearningRate(minLr)

    // val optim = new MomentumGradientAscent()
    //   .setBeta(0.5)
    //   .setLearningRate(lr)
    //   .setShrinkageRate(shrinkageRate)
    //   .setMinLearningRate(minLr)


    // val optim = new ADAM()
    //   .setLearningRate(lr)
    //   .setShrinkageRate(shrinkageRate)
    //   .setMinLearningRate(minLr)
    //   .setBeta1(0.9)
    //   .setBeta2(0.99)

    // val optim = new NesterovGradientAscent()
    //   .setLearningRate(lr)
    //   .setShrinkageRate(shrinkageRate)
    //   .setMinLearningRate(minLr)
      //.setWeightsOptimizer(new RatioWeightTransformation())

    if(batchSize.isDefined){
      optim.setBatchSize(batchSize.get)
    }

    //val optim = new ADAM(learningRate = lr,regularizer= None,beta1=0.9,beta2=0.1).setShrinkageRate(shrinkageRate).setMinLearningRate(minLr)
    var model = GradientBasedGaussianMixture(initialWeights,initialDists.clone,optim)

    //val model = optim.fit(parsedData,k=3)
    
    model.step(parsedData)

    sc.stop()
  }
}

class GeneratePathLogs extends FlatSpec{
  "it" should "generate path logs" in {
    FitTestData.run
    assert(true)
  }
}