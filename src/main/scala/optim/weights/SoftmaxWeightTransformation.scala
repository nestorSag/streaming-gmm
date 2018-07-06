package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, max, min, sum}

import breeze.numerics.{exp, log}
class SoftmaxWeightTransformation extends GMMWeightTransformation {
	
	private val (upperBound,lowerBound) = findBounds
	
	def toSimplex(soft: BDV[Double]): BDV[Double] = {

		val bounded = exp(bound(soft))
		bounded/sum(bounded)
	}

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		val n = posteriors.sum

		posteriors - weights*n
	}

	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		val d = weights.length
		log(weights/weights(d-1))
	}

	def bound(soft: BDV[Double]): BDV[Double] = {
		val offset = -(max(soft) + min(soft))/2
	    val d = soft.length
	    //bound the centered soft weights to avoid under or overflow
	    trim(soft + BDV.ones[Double](d)*offset)

	}

	private def findBounds: (Double,Double) = {
		val bound = {
		  var eps = 1.0
		  while (!math.exp(eps).isInfinite) {
		    eps += 1
		  }
		  eps
		}

		(bound-1,-bound+1)
	}

	private def trim(weights: BDV[Double]): BDV[Double] = {
		for(i <- 1 to weights.length){
		  weights(i-1) = math.max(math.min(weights(i-1),upperBound),lowerBound)
		}
		weights
	}


}