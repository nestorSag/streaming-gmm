package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMRegularizer extends Serializable {

	def gradient(dist:UpdatableMultivariateGaussian): BDM[Double]

	def weightGradient(weight: Double): Double

	def evaluate(dist: UpdatableMultivariateGaussian, weight: Double): Double

}
