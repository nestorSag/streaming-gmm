package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import breeze.numerics.sqrt

/**
  * Compute ADAM algirithm directions. See ''Adam: A Method for Stochastic Optimization. Kingma, Diederik P.; Ba, Jimmy, 2014''

  */
class GMMAdam extends GMMGradientAscent {

/**
  * iteration counter

  */
	var t: Int = 0

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
	
	def setBeta1(beta1: Double): Unit = { 
		require(beta1 > 0 , "beta1 must be positive")
		this.beta1 = beta1
	}

	def getBeta1: Double = { 
		this.beta1
	}

	def setBeta2(beta2: Double): Unit = { 
		require(beta2 > 0 , "beta2 must be positive")
		this.beta2 = beta2
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


	override def direction(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double] = {

		t += 1

		if(!dist.optimUtils.momentum.isDefined){
			dist.optimUtils.initializeMomentum
		}

		if(!dist.optimUtils.adamInfo.isDefined){
			dist.optimUtils.initializeAdamInfo
		}

		val grad = lossGradient(dist, point, w)

		dist.optimUtils.updateMomentum(dist.optimUtils.momentum.get*beta1 + grad*(1.0-beta1))
		
		dist.optimUtils.updateAdamInfo(dist.optimUtils.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * dist.optimUtils.momentum.get /:/ (sqrt(dist.optimUtils.adamInfo.get) + eps)
	}

	override def softWeightsDirection(posteriors: BDV[Double], weights: UpdatableWeights): BDV[Double] = {

		if(!weights.optimUtils.momentum.isDefined){
			weights.optimUtils.initializeMomentum
		}

		if(!weights.optimUtils.adamInfo.isDefined){
			weights.optimUtils.initializeAdamInfo
		}

		val grad = softWeightGradient(posteriors, new BDV(weights.weights))
		
		weights.optimUtils.updateMomentum(weights.optimUtils.momentum.get*beta1 + grad*(1.0-beta1))

		weights.optimUtils.updateAdamInfo(weights.optimUtils.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * weights.optimUtils.momentum.get /:/ (sqrt(weights.optimUtils.adamInfo.get) + eps)

	}

}