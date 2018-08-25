package com.github.gradientgmm.optim

import com.github.gradientgmm.components.Utils

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, max, min, sum}

/**
  * Ratio mapping to fit the weight vector

  * The precise mapping is w_i => w_i/w_last
	

  * Using it is NOT recommended; use the default Softmax transformation instead.
  */
@deprecated("This is an experimental, numerically unstable transformation and should not be used", "gradientgmm >= 1.4")
class RatioWeightTransformation extends WeightsTransformation {

/**
  * machine's epsilon
  */
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

		//val n = posteriors.sum

		(posteriors - weights) /:/ fromSimplex(weights)
	}

	def bound(soft: BDV[Double]): BDV[Double] = {
		val scaleFactor = math.exp(-(max(soft)+min(soft))/2)
	    //bound the centered soft weights to avoid under or overflow
	    trim(soft*scaleFactor)

	}

/**
  * Trim extreme values to avoid over or underflows

  */
  
	private def trim(weights: BDV[Double]): BDV[Double] = {
		for(i <- 1 to weights.length){
			//rescaling by 100 to account for the summation at the denominator of the weight mapping
			// assuming k <= 100
		  weights(i-1) = math.max(math.min(weights(i-1),1e2/eps),eps*1e2)
		}
		weights
	}
}