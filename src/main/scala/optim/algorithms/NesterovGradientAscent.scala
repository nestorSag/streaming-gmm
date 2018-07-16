package com.github.gradientgmm.optim.algorithms

import com.github.gradientgmm.components.AcceleratedGradientUtils

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Optimizer that performs stochastic gradient ascent with Nesterov's correction
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

		val delta = ops.rescale(grad,learningRate)

		val res = ops.sub(
		         delta,
		        ops.rescale(utils.momentum.get,gamma/(1+gamma)))

		utils.updateMomentum(delta)

		res


	}

	override def getWeightsUpdate(current: BDV[Double], grad:BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {

		if(!utils.momentum.isDefined){
			utils.updateMomentum(current)
		}

		val update = toSimplex( (fromSimplex(current) + direction(grad,utils)(vectorOps)) * (1 + gamma))

		utils.updateMomentum(current + utils.momentum.get)

		update

	}


	override def getGaussianUpdate(current: BDM[Double], grad:BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {

		if(!utils.momentum.isDefined){
			utils.updateMomentum(current)
		}

		val update = (current + direction(grad,utils)(matrixOps)) * (1 + gamma)

		utils.updateMomentum(current + utils.momentum.get)

		update

	}

	
}