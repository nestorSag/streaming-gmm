package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, max, min, sum}

class RatioWeightTransformation extends GMMWeightTransformation {
	
	val eps = Utils.EPS

	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		val d = weights.length
		weights/weights(d-1)
	}

	def toSimplex(soft: BDV[Double]): BDV[Double] = {

	    val bounded = bound(soft)
	    bounded/sum(bounded)
  }

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		val n = posteriors.sum

		(posteriors - weights*n) /:/ fromSimplex(weights)
	}

	def bound(soft: BDV[Double]): BDV[Double] = {
		val scaleFactor = math.exp(-(max(soft)+min(soft))/2)
	    //bound the centered soft weights to avoid under or overflow
	    trim(soft*scaleFactor)

	}

	private def trim(weights: BDV[Double]): BDV[Double] = {
		for(i <- 1 to weights.length){
		  weights(i-1) = math.max(math.min(weights(i-1),Double.MaxValue),eps)
		}
		weights
	}
}