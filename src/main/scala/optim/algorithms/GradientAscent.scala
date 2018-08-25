package com.github.gradientgmm.optim

import com.github.gradientgmm.components.AcceleratedGradientUtils

/**
  * Implementation of standard gradient ascent
  */
class GradientAscent extends Optimizer{ 

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {
		grad
	}
	
}