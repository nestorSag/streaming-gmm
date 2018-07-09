package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}


/**
  * A class that wraps necessary object for accelerated gradient descent methods.
  *
  * Intializes and stores the necessary numeric data structures to calculate ascent directions in
  * Momentum gradient descent and the ADAM algorithm. It only initializes such objects when they
  * are called, avoiding storing unecessary information.

  * @tparam T The structure data type, e.g., DenseVector or DenseMatrix
  * @param zeroGenerator Function that initialize the data structures to zero.
  * @param d The data structure dimension(s).
 
  */
abstract class AcceleratedGradientUtils[T <: Any](val zeroGenerator: Int => T, val d: Int) extends Serializable{

/**
  * momentum term data structure
  *  */

  private[gradientgmm] var momentum: Option[T] = None
  
  /**
  * ADAM information data structure
  *
  */

  private[gradientgmm] var adamInfo: Option[T] = None

/**
  * Set the momentum term equal to x. See [[GMMGradientAscent]]
  *
  */

  private[gradientgmm] def updateMomentum(x: T): Unit = {
    momentum = Option(x)
  }

/**
  * Set the ADAM term equal to x. See [[GMMAdam]]
  *
  * @return returns this object
  */

  private[gradientgmm] def updateAdamInfo(x: T): Unit = {
    adamInfo = Option(x)
  }

/**
  * Set momentum to zero
  *
  * @return returns this object
  */

  private[gradientgmm] def initializeMomentum: Unit = {
    momentum = Option(zeroGenerator(d))
  }

/**
  * Set ADAM information to zero
  *
  * @return returns this object
  */

  private[gradientgmm] def initializeAdamInfo: Unit = {
    adamInfo = Option(zeroGenerator(d))
  }

/**
  * Reset (Throw away) the momentum and ADAM terms
  *
  * @return returns this object
  */
  def lighten: this.type = {
    momentum = None
    adamInfo = None
    this
  }
}

/**
  * Implementation of [[AcceleratedGradientUtils]] to allow vector-like acceleration terms.
  *

  * It uses Breeze's DenseVector as the data type
  * @param d Vector length
 
  */

class VectorGradientUtils(
  d: Int) extends AcceleratedGradientUtils[BDV[Double]]((n:Int) => BDV.zeros[Double](n),d)


/**
  * Container of a [[VectorGradientUtils]] instance
  *
 
  */
trait VectorOptimUtils extends Serializable{
  val optimUtils: VectorGradientUtils
}

/**
  * Implementation of [[AcceleratedGradientUtils]] to allow matrix-like acceleration terms.
  *

  * It uses Breeze's DenseMatrix as the data type
  * @param d Square matrix size
 
  */

class MatrixGradientUtils(
  d: Int) extends AcceleratedGradientUtils[BDM[Double]]((n:Int) => BDM.zeros[Double](n,n),d)


/**
  * Container of a [[MatrixGradientUtils]] instance
  *
 
  */
trait MatrixOptimUtils extends Serializable{
  val optimUtils: MatrixGradientUtils
}

