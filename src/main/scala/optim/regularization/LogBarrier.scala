package com.github.gradientgmm.optim.regularization

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Regularization term of the form {{{scale*log(det(cov) - shift)}}}

  */

class LogBarrier extends Regularizer{ 


	private var scale = 1.0

	private var shift = 0.0

	def setScale(scale: Double): this.type = {
		require(scale > 0, "scale must be positive")
		this.scale = scale
		this
	}

	def setShift(shift: Double): this.type = {
		require(shift >= 0, "shift must be nonnegative")
		this.shift = shift
		this
	}

	def getScale = scale

	def getShift = shift

	def weightsGradient(weights: BDV[Double]): BDV[Double] = BDV.zeros[Double](weights.length)

	def gaussianGradient(dist: UpdatableGaussianComponent): BDM[Double] = {

		val lastCol = dist.paramMat(::,dist.paramMat.cols-1)

		// exact calculation when shift >0 can cause numerical overflow if dimensionality is high
		// that is why the calculation differ in this case
		if(shift >0){
			val detS = dist.detSigma*dist.getS
			val paramMat = dist.paramMat
			(paramMat*(detS/(detS - shift*dist.getS)) - lastCol*lastCol.t*(1 + shift/(detS - shift*dist.getS))) * scale
		}else{
			(dist.paramMat - lastCol*lastCol.t) * scale
		}
	}

	def evaluateDist(dist: UpdatableGaussianComponent): Double = {
		if(shift >0){
			scale * (math.log(dist.detSigma - shift) + math.log(dist.getS) - dist.getS)
		}else{
			scale * (dist.logDetSigma + math.log(dist.getS) - dist.getS)
		}
	}

	def evaluateWeights(weights: BDV[Double]): Double = {
		0.0
	}

}