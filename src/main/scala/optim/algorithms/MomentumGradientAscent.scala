package com.github.gradientgmm.optim.algorithms

import com.github.gradientgmm.components.AcceleratedGradientUtils

/**
  * Optimizer that performs stochastic gradient ascent with momentum

  * As formulated in ''Goh, "Why Momentum Really Works", Distill, 2017. http://doi.org/10.23915/distill.00006''
  */

class MomentumGradientAscent extends Optimizer {

/**
  * Inertia parameter
  */
	var beta = 0.5
	
	def setBeta(beta: Double): this.type = { 
		require(beta > 0 & beta < 1, "beta must be in (0,1)")
		this.beta = beta
		this
	}

	def getBeta: Double = { 
		this.beta
	}

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {
		
		if(!utils.momentum.isDefined){
			utils.initializeMomentum
		}
		
		utils.updateMomentum(
			ops.sum(
				ops.rescale(utils.momentum.get,beta),
				grad))

		utils.momentum.get
	}


}