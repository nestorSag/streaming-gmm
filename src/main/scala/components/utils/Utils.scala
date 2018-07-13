package com.github.gradientgmm.components

import breeze.linalg.{DenseVector => BDV}

/**
  * Contains non-specific functions that are used by many classes in the package
 
  */
private[gradientgmm] object Utils extends Serializable{

/**
  * Machine epsilon
 
  */
  val EPS = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

/**
  * Transforms an Array to a Breeze vector
 
  */
  def toBDV(x: Array[Double]): BDV[Double] = {
    new BDV(x)
  }

}