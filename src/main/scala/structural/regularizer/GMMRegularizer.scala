package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Basic functionality for a regularization term.
  * See [[https://en.wikipedia.org/wiki/Regularization_(mathematics)]]
  *
  * 

  */
trait GMMRegularizer extends Serializable {

/**
  * Computes the loss function's gradient w.r.t a component's parameters
  *
  * @param dist Mixture component
  * @return gradient
 
  */
	def gradient(dist:UpdatableGaussianMixtureComponent): BDM[Double]


/**
  * Computes the loss function's gradient w.r.t the current weight vector
  *
  * @param weights current weights vector
  * @return gradient
 
  */
	def weightsGradient(weights: BDV[Double]): BDV[Double]


/**
  * Evaluate regularization term for the current component and corresponding weight
  *
  * @param dist Mixture component
  * @param weight component's weight
  * @return regularization value
 
  */
	def evaluate(dist: UpdatableGaussianMixtureComponent, weight: Double): Double

}
