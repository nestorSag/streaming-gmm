package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseVector => BDV}

class UpdatableWeights(var weights: Array[Double]) extends Serializable with VectorOptimUtils{

  require(checkPositivity(weights), "some weights are negative or equal to zero")
  require(isInSimplex(weights),s"new weights don't sum 1: ${weights.mkString(",")}")

  var simplexErrorTol = 1e-8
  val d = weights.length
  
  val optimUtils = new VectorGradientUtils(d)

  def length: Int = d
  
  def update(newWeights: BDV[Double]): Unit = {
    // recenter soft weights to avoid under or overflow
    val newW = newWeights.toArray
    require(isInSimplex(newW),s"new weights don't sum 1: ${newW.mkString(",")}")
    weights = newW

  }

  def isInSimplex(x: Array[Double]): Boolean = {
    val s = x.sum
    val error = (s-1.0)
    error*error <= simplexErrorTol
  }

  def checkPositivity(x: Array[Double]): Boolean = {
    var allPositive = true
    var i = 0
    while(i < x.length && allPositive){
      if(x(i)<=0){
        allPositive = false
      }
      i += 1
    }
    allPositive
  }

}