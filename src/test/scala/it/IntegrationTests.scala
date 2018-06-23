import org.scalatest.{FlatSpec,BeforeAndAfter}


import streamingGmm.{UpdatableMultivariateGaussian, SGDGMM, GMMGradientAscent}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

// tests for UpdatableMultivariateGaussian class

class SGDGMMTest extends FlatSpec {

  val master = "local[4]"
  val appName = "SGDGMM-test"

  var sc: SparkContext = _
  val conf = new SparkConf().setMaster(master).setAppName(appName)
  sc = new SparkContext(conf)
  
  val errorTol = 1e-8
  val dim = 10
  val k = 5

  val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
  val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()

  val mygmm = SGDGMM(k,new GMMGradientAscent(0.9,None),parsedData)

  val weights = mygmm.getWeights
  val gaussians = mygmm.getGaussians.map{case g => new MultivariateGaussian(SVS.dense(g.getMu.toArray),SMS.dense(dim,dim,g.getSigma.toArray))}

  val sparkgmm = new GaussianMixtureModel(weights,gaussians)

  val x = parsedData.take(1)(0)

  "predict()" should "give same result as spark GMM model for single vector" in {
    assert(sparkgmm.predict(x) - mygmm.predict(x) == 0)
  }

  "predict()" should "give same result as spark GMM model for RDD" in {

    val res = sparkgmm.predict(parsedData).zip(mygmm.predict(parsedData)).map{case (x,y) => (x-y)*(x-y)}.sum

    assert(res == 0)
  }

  "predictSoft()" should "give same result as spark GMM model for single vector" in {
    var v1 = new BDV(sparkgmm.predictSoft(x))
    var v2 = new  BDV(mygmm.predictSoft(x))

    assert(norm(v1-v2)*norm(v1-v2) < errorTol)
  }

  "predictSoft()" should "give same result as spark GMM model for RDD" in {
    val res = sparkgmm.predictSoft(parsedData).zip(mygmm.predictSoft(parsedData)).map{case (a,b) => {
      val x = new BDV(a)
      val y = new BDV(b)
      norm(x-y)*norm(x-y)
      }}.sum

    assert(res == 0)
  }
  // the step() method will be tested in the integration testing stage

  sc.stop()
}




// val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
// val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()

// val optim = new GMMGradientAscent(learningRate = 0.9,regularizer= None)
// var model = SGDGMM(k = 4, optimizer = optim, data = parsedData)

// val initialWeights = model.getWeights
// val initialDists = model.getGaussians


// val initialSparkDists = initialDists.map{ d => new MultivariateGaussian(SVS.dense(d.getMu.toArray),SMS.dense(d.getSigma.rows,d.getSigma.cols,d.getSigma.toArray))}
// var sparkGmm = new GaussianMixtureModel(initialWeights,initialSparkDists)
// var sparkGm = new GaussianMixture().setK(k).setInitialModel(sparkGmm)


// var sparkFittedModel = sparkGm.run(parsedData)
// var sparkFittedSigmas = sparkFittedModel.gaussians.map{case g => new BDM(g.sigma.numRows,g.sigma.numCols,g.sigma.toArray)}
// var sparkFittedMus = sparkFittedModel.gaussians.map{case g => BDV(g.mu.toArray)}
// var sparkFittedWeights = sparkFittedModel.weights

// val optim = new GMMGradientAscent(learningRate = 0.1,regularizer= None)
// //var fittedModel = SGDGMM(k,parsedData.take(1)(0).size,optim).setmaxGradientIters(100).setWeightLearningRate(0.7)
// var fittedModel = SGDGMM(initialWeights,initialDists,optim).setmaxGradientIters(100).setWeightLearningRate(0.07)
// fittedModel.learningRateShrinkage = 1.0
// fittedModel.step(parsedData)


// var sgdFittedSigmas = fittedModel.getGaussians.map{case g => g.getSigma}
// var sgdFittedMus = fittedModel.getGaussians.map{case g => g.getMu}
// var sgdFittedWeights = fittedModel.getWeights