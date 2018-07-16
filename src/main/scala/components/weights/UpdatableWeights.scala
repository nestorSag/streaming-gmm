package com.github.gradientgmm.components

import breeze.linalg.{DenseVector => BDV}


/**
  * Wrapper class for the weights vector.

  * It includes functionality to check the simplex constraints and perofrm accelerated gradient descent.
  * see [[https://en.wikipedia.org/wiki/Simplex]]
  
  * @param weights weight vector
  */

class UpdatableWeights(var weights: Array[Double]) extends Serializable with VectorParamUpdate{

  require(isPositive(weights), "some weights are negative or equal to zero")
  require(isInSimplex(weights),s"new weights don't sum 1: ${weights.mkString(",")}")

/**
  * Allowed deviation from 1 of the weight vector's sum
 
  */
  var simplexErrorTol = 1e-8

/**
  * weight vector dimensionality
 
  */
  val d = weights.length

/**
  * accelerated gradient descent utilities. See [[AcceleratedGradientUtils]]
 
  */
  val optimUtils = new VectorGradientUtils(d)

/**
  * returns weigh vector dimensionality
 
  */
  def length: Int = d

/**
  * Replaces the weight vector  with a new one, checking for correctness beforehand
  
  * @param newWeights new weight vector
  */
  def update(newParam: BDV[Double]): Unit = {
    // recenter soft weights to avoid under or overflow
    val newW = newParam.toArray
    require(isInSimplex(newW),s"new weights don't sum 1: ${newW.mkString(",")}")
    weights = newW

  }

/**
  * Checks whether the given vector is within acceptable distance to the weight simplex
  
  * @param w weight vector
  */
  def isInSimplex(w: Array[Double]): Boolean = {
    val s = w.sum
    val error = (s-1.0)
    error*error <= simplexErrorTol
  }

/**
  * Checks that the initial weight vector does not contains any single negative or null weight element, in which case it returns {{{false}}}
  * @param w weight vector

  */
  def isPositive(w: Array[Double]): Boolean = {
    var allPositive = true
    var i = 0
    while(i < w.length && allPositive){
      if(w(i)<=0){
        allPositive = false
      }
      i += 1
    }
    allPositive
  }

}