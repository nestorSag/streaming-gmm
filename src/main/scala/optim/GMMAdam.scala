package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import breeze.numerics.sqrt

class GMMAdam extends GMMGradientAscent {

	var t: Int = 0 //timestep
	var eps = 1e-8

	var beta1 = 0.5
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

	def reset: Unit = {
		t = 0
	}

	def setEps(x: Double): Unit = {
		require(x>=0,"x should be nonnegative")
		eps = x
	}


	override def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		t += 1

		if(!dist.momentum.isDefined){
			dist.initializeMomentum
		}

		if(!dist.adamInfo.isDefined){
			dist.initializeAdamInfo
		}

		val grad = lossGradient(dist, sampleInfo)

		dist.updateMomentum(dist.momentum.get*beta1 + grad*(1.0-beta1))
		
		dist.updateAdamInfo(dist.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * dist.momentum.get /:/ (sqrt(dist.adamInfo.get) + eps)
	}

	override def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double] = {

		if(!weights.momentum.isDefined){
			weights.initializeMomentum
		}

		if(!weights.adamInfo.isDefined){
			weights.initializeAdamInfo
		}

		val grad = softWeightGradient(posteriors, new BDV(weights.weights))
		
		weights.updateMomentum(weights.momentum.get*beta1 + grad*(1.0-beta1))

		weights.updateAdamInfo(weights.adamInfo.get*beta2 + (grad *:* grad)*(1.0-beta2))

		val alpha_t = math.sqrt(1.0 - math.pow(beta2,t))/(1.0 - math.pow(beta1,t))

		alpha_t * weights.momentum.get /:/ (sqrt(weights.adamInfo.get) + eps)

	}

}