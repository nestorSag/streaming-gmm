package com.github.gradientgmm.optim

import com.github.gradientgmm.components.AcceleratedGradientUtils

/**
  * Optimizer that performs Stochastic Gradient Ascent
  */
class GradientAscent extends Optimizer{ 

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {
		grad
	}
	
}