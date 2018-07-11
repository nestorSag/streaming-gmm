package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import breeze.numerics.sqrt

/**
  * Compute ADAM algirithm directions. See ''Adam: A Method for Stochastic Optimization. Kingma, Diederik P.; Ba, Jimmy, 2014''
  */
class GMMAdam extends GMMOptimizer {

/**
  * iteration counter
  */
	var t: Int = 1

/**
  * offset term to avoid division by zero in the main direction calculations
  */
	var eps = 1e-8

/**
  * Exponential smoothing parameter for the first raw moment estimator 
  */
	var beta1 = 0.5

/**
  * Exponential smoothing parameter for the second raw moment estimator 
  */
	var beta2 = 0.1
	
	def setBeta1(beta1: Double): this.type = { 
		require(beta1 > 0 , "beta1 must be positive")
		this.beta1 = beta1
		this
	}

	def getBeta1: Double = { 
		this.beta1
	}

	def setBeta2(beta2: Double): this.type = { 
		require(beta2 > 0 , "beta2 must be positive")
		this.beta2 = beta2
		this
	}

	def getBeta2: Double = { 
		this.beta2
	}

/**
  * Reset iterator counter
  */
	def reset: Unit = {
		t = 0
	}

	def setEps(x: Double): Unit = {
		require(x>=0,"x should be nonnegative")
		eps = x
	}

	def getEps: Double = eps


	def direction(grad: BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		t += 1

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
		}

		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		utils.updateMomentum(utils.momentum.get*beta1 + grad*(1.0-beta1))
		
		utils.updateAdamInfo(utils.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * utils.momentum.get /:/ (sqrt(utils.adamInfo.get) + eps)
	}

	override def weightsDirection(posteriors: BDV[Double], weights: UpdatableWeights): BDV[Double] = {

		if(!weights.optimUtils.momentum.isDefined){
			weights.optimUtils.initializeMomentum
		}

		if(!weights.optimUtils.adamInfo.isDefined){
			weights.optimUtils.initializeAdamInfo
		}

		val grad = weightsGradient(posteriors, new BDV(weights.weights))
		
		weights.optimUtils.updateMomentum(weights.optimUtils.momentum.get*beta1 + grad*(1.0-beta1))

		weights.optimUtils.updateAdamInfo(weights.optimUtils.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * weights.optimUtils.momentum.get /:/ (sqrt(weights.optimUtils.adamInfo.get) + eps)

	}

}