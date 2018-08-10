package com.github.gradientgmm.optim

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Basic functionality for a [[https://en.wikipedia.org/wiki/Regularization_(mathematics) regularization]] term.
  * 
  *
  * 

  */
trait Regularizer extends Serializable {

/**
  * Computes the loss function's gradient w.r.t a component's parameters
  *
  * @param dist Mixture component
  * @return gradient
 
  */
	def gaussianGradient(dist:UpdatableGaussianComponent): BDM[Double]


/**
  * Computes the loss function's gradient w.r.t the current weight vector
  *
  * @param weights current weights vector
  * @return gradient
 
  */
	def weightsGradient(weights: BDV[Double]): BDV[Double]


/**
  * Evaluate regularization term for a mixture component
  *
  * @param dist Mixture component
  * @return regularization value
 
  */
	def evaluateDist(dist: UpdatableGaussianComponent): Double

/**
  * Evaluate regularization term for the weights vector
  *
  * @param weights model weights vector
  * @return regularization value
 
  */
  def evaluateWeights(weights: BDV[Double]): Double

}
