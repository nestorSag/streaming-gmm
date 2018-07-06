package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMRegularizer extends Serializable {

	def gradient(dist:UpdatableMultivariateGaussian): BDM[Double]

	def softWeightsGradient(weights: BDV[Double]): BDV[Double]

	def evaluate(dist: UpdatableMultivariateGaussian, weight: Double): Double

}
