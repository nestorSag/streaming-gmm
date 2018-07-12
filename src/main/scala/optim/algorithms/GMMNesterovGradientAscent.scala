package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Compute gradient ascent directions with momentum
  */

class GMMNesterovGradientAscent extends GMMOptimizer {

/**
  * correction parameter
  */
	var gamma = 0.5
	
	def setGamma(gamma: Double): this.type = { 
		require(gamma > 0 , "gamma must be positive")
		this.gamma = gamma
		this
	}

	def getGamma: Double = { 
		this.gamma
	}

	def gaussianDirection(grad: BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		// if(!utils.momentum.isDefined){
		// 	utils.initializeMomentum
		// }
		
		// utils.updateMomentum(utils.momentum.get*gamma + grad)

		// utils.momentum.get
		grad
	}

	def weightsDirection(grad: BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {

		// if(!utils.momentum.isDefined){
		// 	utils.initializeMomentum
		// }
		
		// utils.updateMomentum(utils.momentum.get*gamma + grad)

		// utils.momentum.get
		grad

	}

	override def getGaussianUpdate(current: BDM[Double], grad: BDM[Double],  utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
			utils.updateMomentum(current)
		}

		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		utils.updateAdamInfo(current + grad * learningRate)

		val update = current + (utils.adamInfo.get - utils.momentum.get) * gamma

		utils.updateMomentum(utils.adamInfo.get)

		update
	}

	override def getWeightsUpdate(current: BDV[Double], grad: BDV[Double],  utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
			utils.updateMomentum(current)
		}

		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		utils.updateAdamInfo(current + grad * learningRate)

		val update = current + (utils.adamInfo.get - utils.momentum.get) * gamma

		utils.updateMomentum(utils.adamInfo.get)

		update
	}

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(implicit ops: ParameterOperations[A]): A = {
		grad
	}

	override def getUpdate[A](current: A, grad:A, utils: AcceleratedGradientUtils[A])(implicit ops: ParameterOperations[A]): A = {
		
		if(!utils.momentum.isDefined){
			utils.initializeMomentum
			utils.updateMomentum(current)
		}

		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		utils.updateAdamInfo(
			ops.sum(
				current,
				ops.rescale(grad,learningRate)))

		val update = ops.sum(
			current,
			ops.rescale(ops.sub(utils.adamInfo.get,utils.momentum.get),gamma))

		utils.updateMomentum(utils.adamInfo.get)

		update
	}
	
}