package edu.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class SoftmaxWeightTransformation extends GMMWeightTransformation {

	def weightToSoft(weights: Array[Double]): BDV[Double] = {
    	new BDV(weights.map{case w => math.log(w/weights.last)})
	}


	def softToWeight(soft: BDV[Double]): Array[Double] = {

		val expsoft = soft.toArray.map{ case z => math.exp(z)}

		expsoft.map{case w => w/expsoft.sum}
	}

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		val n = posteriors.sum

		posteriors - weights*n
	}

}