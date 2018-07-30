package com.github.gradientgmm.optim.regularization

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Regularization term of the form scale*log(det(cov) - shift)

  */

class LogBarrier extends Regularizer{ 


	private var scale = 1.0

	def setScale(scale: Double): this.type = {
		require(scale >= 0.5, "scale must be at least 0.5 to avoid covariance singularities")

		this.scale = scale
		this
	}

	def getScale = scale

	def weightsGradient(weights: BDV[Double]): BDV[Double] = BDV.zeros[Double](weights.length)

	def gaussianGradient(dist: UpdatableGaussianComponent): BDM[Double] = {

		val lastCol = dist.paramMat(::,dist.paramMat.cols-1)

		(dist.paramMat - lastCol*lastCol.t) * scale
	}

	def evaluateDist(dist: UpdatableGaussianComponent): Double = {

		scale * (dist.logDetSigma + math.log(dist.getS) - dist.getS)
	}

	def evaluateWeights(weights: BDV[Double]): Double = {
		0.0
	}

}