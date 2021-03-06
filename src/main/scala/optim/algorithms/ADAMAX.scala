package com.github.gradientgmm.optim

import com.github.gradientgmm.components.AcceleratedGradientUtils

/**
  * Implementation of ADAMAX. See ''Adam: A Method for Stochastic Optimization. Kingma, Diederik P.; Ba, Jimmy, 2014''
  
  * Using it is NOT recommended; you should use SGD or its accelerated versions instead.
  */
@deprecated("ADAMAX can be unstable for GMM problems and should not be used", "gradientgmm >= 1.4")
class ADAMAX extends Optimizer {

/**
  * iteration counter
  */
	var t: Double = 1.0

/**
  * Exponential smoothing parameter for the first raw moment estimator 
  */
	var beta1 = 0.9

	private var eps = 1e-8 //needed because last weight gradient is always zero

/**
  * Exponential smoothing parameter for the second raw moment estimator 
  */
	var beta2 = 0.999
	
	def setBeta1(beta1: Double): this.type = { 
		require(beta1 >= 0 & beta1 < 1, "beta1 must be in [0,1)")
		this.beta1 = beta1
		this
	}

	def getBeta1: Double = { 
		this.beta1
	}

	def setBeta2(beta2: Double): this.type = { 
		require(beta2 >= 0 & beta2 < 1 , "beta2 must be in [0,1)")
		this.beta2 = beta2
		this
	}

	def getBeta2: Double = { 
		this.beta2
	}

/**
  * Reset iterator counter
  */
	def reset: Unit = {
		t = 0.0
	}

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A = {

	t += 1.0

	if(!utils.momentum.isDefined){
		utils.initializeMomentum
	}

	if(!utils.adamInfo.isDefined){
		utils.initializeAdamInfo
	}
	
	utils.setMomentum(
		ops.sum(
			ops.rescale(utils.momentum.get,beta1), 
			grad))

	//the following lines compute a maximum through absolute values
	// max(a,b) = a + b + |b-a|
	// because max() is not implementd for elementwise matrix comparison in breeze

	// | |g| - beta2*u |
	val aux1 = ops.ewAbs(
		ops.sub(
			ops.ewAbs(grad),
			ops.rescale(utils.adamInfo.get,beta2)))


	//  |g| + beta2*u 
	val aux2 = ops.sum(
		ops.ewAbs(grad),
		ops.rescale(utils.adamInfo.get,beta2))

	//max(beta2*u,|g|) = 0.5 * (|g| + beta2*u + |beta2*u - |g||) = 0.5*(aux1+aux2)
	utils.setAdamInfo(
		ops.rescale(
			ops.sum(aux1,aux2),
			0.5))

	val alpha_t = 1.0/(1.0 - math.pow(beta1,t))

	ops.rescale(
		ops.ewDiv(
			utils.momentum.get,
			ops.sumScalar(utils.adamInfo.get,eps)),
		alpha_t)


	}

}