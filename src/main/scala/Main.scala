import streamingGmm.{UpdatableMultivariateGaussian, GradientBasedGaussianMixture, GMMGradientAscent}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

object Main extends App {

	var sc : SparkContext = _

	val conf = new SparkConf().setAppName("GradientBasedGaussianMixture-test").setMaster("local[4]")
	sc = new SparkContext(conf)

	val errorTol = 1e-8
	val k = 3

	val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
	val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()
	val d = parsedData.take(1)(0).size

	val mygmm = GradientBasedGaussianMixture(k,new GMMGradientAscent(0.9,None),parsedData)

	val weights = mygmm.getWeights
	val gaussians = mygmm.getGaussians.map{
	case g => new MultivariateGaussian(
	SVS.dense(g.getMu.toArray),
	SMS.dense(g.getSigma.rows,g.getSigma.cols,g.getSigma.toArray))}

	val sparkgmm = new GaussianMixtureModel(weights,gaussians)

	val x = parsedData.take(1)(0)

	val initialWeights = (1 to k).map{case x => 1.0/k}.toArray

	val means = Array(BDV(-1.0,0.0),BDV(0.0,-1.0),BDV(1.0,0.0))
	val covs = (1 to k).map{case k => BDM.eye[Double](d)}.toArray
	val initialDists = means.zip(covs).map{case (m,s) => UpdatableMultivariateGaussian(m,s)}

	val optim = new GMMGradientAscent(learningRate = 0.01,regularizer= None).setShrinkageRate(1.0)
	//val optim = new GMMMomentumGradientAscent(learningRate = 0.9,regularizer= None,decayRate=0.9).setShrinkageRate(0.9).setMinLearningRate(0)
	var fittedModel = GradientBasedGaussianMixture(initialWeights,initialDists.clone,optim).setmaxGradientIters(100)//.setBatchSize(25)
	//var fittedModel = GradientBasedGaussianMixture(3,parsedData.take(1)(0).size,optim).setmaxGradientIters(100)//.setBatchSize(50)
	fittedModel.step(parsedData)

	sc.stop()
}