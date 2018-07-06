package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMOptimizer extends Serializable {

	val regularizer: Option[GMMRegularizer] = None
	val weightOptimizer: GMMWeightTransformation = new SoftmaxWeightTransformation()
	private[gradientgmm] var learningRate: Double = 0.9
	private[gradientgmm] var minLearningRate: Double = 1e-2
	private[gradientgmm] var shrinkageRate: Double = 0.95

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
	
	def penaltyValue(dist: UpdatableMultivariateGaussian,weight: Double): Double

	def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double]

	def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double]

}