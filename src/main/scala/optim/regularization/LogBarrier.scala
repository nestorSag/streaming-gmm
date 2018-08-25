package com.github.gradientgmm.optim

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Regularization term of the form scale*log(det(cov))

  */

class LogBarrier extends Regularizer{ 


	private var scale = 1.0

	def setScale(scale: Double): this.type = {
		require(scale >= 1.0, "scale must be at least 1.0 to avoid covariance singularities")

		this.scale = scale
		this
	}

	def getScale = scale

// *
//   * Shift substracted from det(cov) inside the logarithmic barrier. The purpose of this is to 
//   * account for anomalous data that could induce a singular covariance matrix (data that 
//   * is embeded in a subspace). However, its use is not recommended for high-dimensional data, since
//   * it involved calculating the determinant of the covariance matrix, which can cause numerical
//   * overflow because it is exponential in the matrix size.

  
// 	def setShift(shift: Double): this.type = {
// 		require(shift >= 0.0, "shift must be positive")

// 		this.shift = shift
// 		this
// 	}

// 	def getShift = shift

	def weightsGradient(weights: BDV[Double]): BDV[Double] = BDV.zeros[Double](weights.length)

	def gaussianGradient(dist: UpdatableGaussianComponent): BDM[Double] = {

		val lastCol = dist.paramBlockMatrix(::,dist.paramBlockMatrix.cols-1)

		// exact calculation when shift >0 can cause numerical overflow if dimensionality is high
		// that is why the calculation differ in this case
		// if(shift >0){
		// 	val detS = dist.detSigma*dist.getS
		// 	(dist.paramBlockMatrix*(detS/(detS - shift*dist.getS)) - lastCol*lastCol.t*(1 + shift/(detS - shift*dist.getS))) * scale
		// }else{
		// 	(dist.paramBlockMatrix - lastCol*lastCol.t) * scale
		// }

		(dist.paramBlockMatrix - lastCol*lastCol.t) * scale

	}

	def evaluateDist(dist: UpdatableGaussianComponent): Double = {

		// if(shift >0){
		// 	scale * math.log(dist.detSigma - shift) + math.log(dist.getS) - dist.getS
		// }else{
		// 	scale * (dist.logDetSigma + math.log(dist.getS) - dist.getS)
		// }

		scale * (dist.logDetSigma + math.log(dist.getS) - dist.getS)
	}

	def evaluateWeights(weights: BDV[Double]): Double = {
		0.0
	}

}