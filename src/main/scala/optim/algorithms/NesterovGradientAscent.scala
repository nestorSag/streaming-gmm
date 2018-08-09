package com.github.gradientgmm.optim.algorithms

import com.github.gradientgmm.components.AcceleratedGradientUtils

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Optimizer that performs stochastic gradient ascent with Nesterov's correction
  */

class NesterovGradientAscent extends Optimizer {

/**
  * Inertia parameter
  */
	var gamma = 0.6
	
	def setGamma(gamma: Double): this.type = { 
		require(gamma > 0 & gamma < 1, "gamma must be in (0,1)")
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

	override def getUpdate[A](current: A, grad:A, utils: AcceleratedGradientUtils[A])(implicit ops: ParameterOperations[A]): A = {
    
        if(!utils.momentum.isDefined){
			utils.updateMomentum(current)
		}

		//val update = (current + direction(grad,utils)(ops)) * (1 + gamma)
		val update = ops.rescale(ops.sum(current,direction(grad,utils)(ops)),1+gamma)

		utils.updateMomentum(ops.sum(current,utils.momentum.get))

		update

  }

	
}