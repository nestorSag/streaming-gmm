package edu.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMWeightTransformation extends Serializable {

	def weightToSoft(weights: Array[Double]): BDV[Double]

	def softToWeight(soft: BDV[Double]): Array[Double]

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double]
}