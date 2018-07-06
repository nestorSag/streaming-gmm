package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMWeightTransformation extends Serializable {

	def fromSimplex(weights: BDV[Double]): BDV[Double]

	def toSimplex(soft: BDV[Double]): BDV[Double]

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double]

	def bound(soft: BDV[Double]): BDV[Double]
}