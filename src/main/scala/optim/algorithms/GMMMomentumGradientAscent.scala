package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class GMMMomentumGradientAscent extends GMMGradientAscent {

	var beta = 0.5
	
	def setBeta(beta: Double): this.type = { 
		require(beta > 0 , "beta must be positive")
		this.beta = beta
		this
	}

	def getBeta: Double = { 
		this.beta
	}

	override def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		if(!dist.momentum.isDefined){
			dist.initializeMomentum
		}

		val grad = lossGradient(dist, sampleInfo)
		
		dist.updateMomentum(dist.momentum.get*beta + grad)

		dist.momentum.get
	}

	override def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double] = {

		if(!weights.momentum.isDefined){
			weights.initializeMomentum
		}

		val grad = softWeightGradient(posteriors, new BDV(weights.weights))
		
		weights.updateMomentum(weights.momentum.get*beta + grad)

		weights.momentum.get

	}

}