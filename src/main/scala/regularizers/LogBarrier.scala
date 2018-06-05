package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class LogBarrier(val shift: Double = 0) extends GMMRegularizer{ 

	require(shift >= 0, "shift must be nonnegative")

	def weightGradient(weight: Double): Double = 0

	def gradient(dist: UpdatableMultivariateGaussian): BDM[Double] = {
		val detS = dist.detSigma*dist.getS

		val paramMat = dist.paramMat
		val lastCol = paramMat(::,paramMat.cols-1)
		paramMat*(detS/(detS - shift*dist.getS)) - lastCol*lastCol.t*(1 + shift/(detS - shift*dist.getS))
	}

	def evaluate(dist: UpdatableMultivariateGaussian, weight: Double): Double = {
		math.log(dist.detSigma - shift) + math.log(dist.getS) - dist.getS
	}

}