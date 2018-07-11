package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Compute gradient ascent directions with momentum
  */

class GMMNesterovGradientAscent extends GMMOptimizer {

/**
  * exponential smoothing parameter. See ''Goh, "Why Momentum Really Works", Distill, 2017. http://doi.org/10.23915/distill.00006''
  */
	var beta = 0.5
	
	def setBeta(beta: Double): this.type = { 
		require(beta > 0 , "beta must be positive")
		this.beta = beta
		this
	}

	def getBeta: Double = { 
		this.beta
	}

	def direction(grad: BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
		}
		
		utils.updateMomentum(utils.momentum.get*beta + grad)

		utils.momentum.get
	}

	def weightsDirection(posteriors: BDV[Double], weights: UpdatableWeights): BDV[Double] = {

		if(!weights.optimUtils.momentum.isDefined){
			weights.optimUtils.initializeMomentum
		}

		val grad = weightsGradient(posteriors, new BDV(weights.weights))
		
		weights.optimUtils.updateMomentum(weights.optimUtils.momentum.get*beta + grad)

		weights.optimUtils.momentum.get

	}

}