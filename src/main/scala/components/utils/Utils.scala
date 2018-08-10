package com.github.gradientgmm.components

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

/**
  * Contains non-specific functions that are used by many classes in the package
 
  */
object Utils extends Serializable{

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

/**
  * Build a symmetric matrix from an array that represents an upper triangular matrix
  *
  * @param x upper triangular matrix array
 
  */

  def completeMatrix(x: Array[Double]): BDM[Double] = {

    // get size
    val d = math.sqrt(x.length).toInt

    //convert to matrix
    val mat = new BDM(d,d,x)
    //fill
    mat += mat.t
    //adjust diagonal elements
    var i = 0
    while(i < d){
      mat(i,i) /= 2
      i+=1
    }

    mat

  }

}