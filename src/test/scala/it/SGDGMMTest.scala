import org.scalatest.FlatSpec


import streamingGmm.{UpdatableMultivariateGaussian, SGDGMM, GMMGradientAscent}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

class SGDGMMIntegratiionTest extends FlatSpec {

	val k = 4 //number of clusters
	val errorTol = 1e-8

	val master = "local[4]"
	val appName = "SGDGMM-test"

	var sc: SparkContext = _
	val conf = new SparkConf().setMaster(master).setAppName(appName)
	sc = new SparkContext(conf)


	val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
	val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()

	val optim = new GMMGradientAscent(learningRate = 0.9,regularizer= None)
	var model = SGDGMM(k = 4, optimizer = optim, data = parsedData)

	val initialWeights = model.getWeights
	val initialDists = model.getGaussians


	val initialSparkDists = initialDists.map{ d => new MultivariateGaussian(SVS.dense(d.getMu.toArray),SMS.dense(d.getSigma.rows,d.getSigma.cols,d.getSigma.toArray))}
	var sparkGmm = new GaussianMixtureModel(initialWeights,initialSparkDists)
	var sparkGm = new GaussianMixture().setK(k).setInitialModel(sparkGmm)

	"model and sparkGmm" should "have the same initial parameters" in {

		// allowing small rounding errors
		var frobNormSum = initialDists.map{
			case g => g.getSigma}.zip(sparkGmm.gaussians.map{
				case g => new BDM(g.sigma.numRows,g.sigma.numCols,g.sigma.toArray)}).map{case (a,b) => trace((a-b)*(a-b))}.sum

		assert(sum < errorTol)

		// allowing small rounding errors
		var vNormSum = initialDists.map{
			case g => g.getMu}.zip(sparkGmm.gaussians.map{
				case g => BDV(g.mu.toArray)}).map{case (a,b) => norm(a-b)}.sum // doing this due to rounding errors

		assert(vNormSum < errorTol)

		assert(initialWeights == sparkGmm.weights)
	}

	var sparkFittedModel = sparkGm.run(parsedData)

	var sparkFittedSigmas = sparkFittedModel.gaussians.map{case g => new BDM(g.sigma.numRows,g.sigma.numCols,g.sigma.toArray)}
	
	var sparkFittedMus = sparkFittedModel.gaussians.map{case g => BDV(g.mu.toArray)}

	var sparkFittedWeights = sparkFittedModel.weights


	var fittedModel = SGDGMM(initialWeights,initialDists,optim).setmaxGradientIters(10)


	var sgdFittedSigmas = fittedModel.getGaussians.map{case g => g.getSigma}
	
	var sgdFittedMus = fittedModel.getGaussians.map{case g => g.getMu}

	var sgdFittedWeights = fittedModel.getWeights

}