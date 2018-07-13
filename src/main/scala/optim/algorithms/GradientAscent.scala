package com.github.nestorsag.gradientgmm.optim.algorithms

import com.github.nestorsag.gradientgmm.components.AcceleratedGradientUtils

/**
  * Optimizer that performs stochastic gradient ascent
  */
class GradientAscent extends Optimizer{ 

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {
		grad
	}
	
}