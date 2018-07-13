package com.github.nestorsag.gradientgmm.optim.weights

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, max, min, sum}

import breeze.numerics.{exp, log}

/**
  * Implements a Softmax mapping to optimize the weight vector

  *The precise mapping is {{{w_i => log(w_i/w_last)}}} and is an implementation of the procedure 
  * described in ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * (see [[https://arxiv.org/abs/1706.03267]]).

  */
class SoftmaxWeightTransformation extends GMMWeightTransformation {
	
	/**
  * upper and lower bounds for allowed values before applying {{{toSimplex}}}

  */
	private val (upperBound,lowerBound) = findBounds
	
	def toSimplex(soft: BDV[Double]): BDV[Double] = {

		val bounded = exp(bound(soft))
		bounded/sum(bounded)
	}

	def gradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		//val n = posteriors.sum

		posteriors - weights
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

/**
  * Use the machine's epsilon to find maximum and minimum allowed values before applying {{{toSimplex}}}

  */
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

/**
  * Trim extreme values to avoid over or underflows

  */
	private def trim(weights: BDV[Double]): BDV[Double] = {
		for(i <- 1 to weights.length){
		  weights(i-1) = math.max(math.min(weights(i-1),upperBound),lowerBound)
		}
		weights
	}


}