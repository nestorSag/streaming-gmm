package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Compute gradient ascent directions with momentum
  */

class NesterovGradientAscent extends Optimizer {

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

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {

		ops.sub(
		  ops.rescale(grad,learningRate),
		  ops.rescale(utils.momentum.get,gamma/(1+gamma)))

	}

	override def getWeightsUpdate(current: BDV[Double], grad:BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {
		
		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
			utils.updateMomentum(current)
		}

		utils.updateAdamInfo(fromSimplex(current) + grad * learningRate)

		val update = toSimplex( (fromSimplex(current) + direction(grad,utils)(vectorOps)) * (1 + gamma))

		utils.updateMomentum(utils.adamInfo.get)

		update

	}


	override def getGaussianUpdate(current: BDM[Double], grad:BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {
		
		if(!utils.adamInfo.isDefined){
			utils.initializeAdamInfo
		}

		if(!utils.momentum.isDefined){
			utils.initializeMomentum
			utils.updateMomentum(current)
		}

		utils.updateAdamInfo(current + grad * learningRate)

		val update = (current + direction(grad,utils)(matrixOps)) * (1 + gamma)

		utils.updateMomentum(utils.adamInfo.get)

		update

	}

	
}