package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

trait GMMOptimizer extends Serializable{

	private[gradientgmm] val regularizer: Option[GMMRegularizer] = None
	private[gradientgmm] val weightOptimizer: GMMWeightTransformation = new SoftmaxWeightTransformation()
	private[gradientgmm] var learningRate: Double = 0.9
	private[gradientgmm] var minLearningRate: Double = 1e-2
	private[gradientgmm] var shrinkageRate: Double = 0.95
	private[gradientgmm] var batchSize: Option[Int] = None
	private[gradientgmm] var convergenceTol: Double = 1e-6
	private[gradientgmm] var maxIter: Int = 100

	def setLearningRate(learningRate: Double): this.type = { 
		require(learningRate > 0 , "learning rate must be positive")
		this.learningRate = learningRate
		this
	}

	def getLearningRate: Double = { 
		this.learningRate
	}

	def setMinLearningRate(m: Double): this.type = { 
		require(m >= 0, "minLearningRate rate must be in positive")
		minLearningRate = m
		this
	}

	def getMinLearningRate: Double = { 
		minLearningRate
	}

	def setShrinkageRate(s: Double): this.type = { 
		require(s > 0 &&  s <= 1.0, "learning rate must be in (0,1]")
		shrinkageRate = s
		this
	}

	def getBatchSize: Option[Int] = batchSize

	def setBatchSize(n: Int): this.type = {
		require(n>0,"n must be a positive integer")
		batchSize = Option(n)
		this
	}

	def getShrinkageRate: Double = { 
		shrinkageRate
	}

	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.fromSimplex(weights)
	}

	def toSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.toSimplex(weights)
	}

	def getConvergenceTol: Double = convergenceTol

	def setConvergenceTol(x: Double): this.type = {
		require(x>0,"convergenceTol must be positive")
		convergenceTol = x
		this
	}


	def setMaxIter(m: Int): this.type = {
		require(m > 0 ,s"maxIter needs to be a positive integer; got ${m}")
		maxIter = m
		this
	}

	def getMaxIter: Int = {
		maxIter
	}
	
	def penaltyValue(dist: UpdatableMultivariateGaussian,weight: Double): Double

	def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double]

	def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double]

	def fit(data: RDD[SV], k: Int, startingSampleSize: Int, kMeansIters: Int, seed: Int): GradientBasedGaussianMixture

}