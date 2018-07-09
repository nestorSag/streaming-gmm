package com.github.nestorsag.gradientgmm

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

	override def direction(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double] = {

		if(!dist.optimUtils.momentum.isDefined){
			dist.optimUtils.initializeMomentum
		}

		val grad = lossGradient(dist, point, w)
		
		dist.optimUtils.updateMomentum(dist.optimUtils.momentum.get*beta + grad)

		dist.optimUtils.momentum.get
	}

	override def softWeightsDirection(posteriors: BDV[Double], weights: UpdatableWeights): BDV[Double] = {

		if(!weights.optimUtils.momentum.isDefined){
			weights.optimUtils.initializeMomentum
		}

		val grad = softWeightGradient(posteriors, new BDV(weights.weights))
		
		weights.optimUtils.updateMomentum(weights.optimUtils.momentum.get*beta + grad)

		weights.optimUtils.momentum.get

	}

}