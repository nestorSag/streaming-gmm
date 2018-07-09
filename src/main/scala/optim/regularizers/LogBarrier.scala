package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class LogBarrier extends GMMRegularizer{ 

	var scale = 1.0
	var shift = 0.0

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

	def softWeightsGradient(weights: BDV[Double]): BDV[Double] = BDV.zeros[Double](weights.length)

	def gradient(dist: UpdatableGConcaveGaussian): BDM[Double] = {

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

	def evaluate(dist: UpdatableGConcaveGaussian, weight: Double): Double = {
		scale * (evaluateGaussian(dist) + evaluateWeight(weight))
	}

	private def evaluateGaussian(dist:UpdatableGConcaveGaussian): Double = {

		if(shift >0){
			math.log(dist.detSigma - shift) + math.log(dist.getS) - dist.getS
		}else{
			dist.logDetSigma + math.log(dist.getS) - dist.getS
		}
	}

	private def evaluateWeight(weights: Double): Double = {
		0.0
	}

}