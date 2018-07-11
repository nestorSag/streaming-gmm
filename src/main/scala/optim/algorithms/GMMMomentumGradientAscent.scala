package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Compute gradient ascent directions with momentum
  */

class GMMMomentumGradientAscent extends GMMOptimizer {

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


	def gaussianDirection(grad: BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
		}
		
		utils.updateMomentum(utils.momentum.get*beta + grad)

		utils.momentum.get
	}

	def weightsDirection(grad: BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
		}
		
		utils.updateMomentum(utils.momentum.get*beta + grad)

		utils.momentum.get

	}

	// override def direction[T <: {def * : Double => T; def + : T =>T}](grad: T, utils: AcceleratedGradientUtils[T]): T = {

	// 	if(!utils.momentum.isDefined){
	// 		utils.initializeMomentum
	// 	}
		
	// 	utils.updateMomentum(utils.momentum.get*beta + grad)

	// 	utils.momentum.get
	// }

	//def getUpdate[T <: {def * : Double => T; def + : T =>T}](current: T, grad: T, utils: AcceleratedGradientUtils[T]): T = 
	//{
	//	current + direction(grad,utils) * learningRate
	//}


}