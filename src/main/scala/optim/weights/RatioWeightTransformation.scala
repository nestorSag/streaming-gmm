package edu.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class RatioWeightTransformation extends GMMWeightTransformation {

	def weightToSoft(weights: Array[Double]): BDV[Double] = {
    	new BDV(weights.map{case w => w/weights.last})
	}

	def softToWeight(soft: BDV[Double]): Array[Double] = {

	    val softarray = soft.toArray

	    softarray.map{case z => z/softarray.sum}
  }

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		val n = posteriors.sum

		(posteriors - weights*n) /:/ weightToSoft(weights.toArray)
	}
}