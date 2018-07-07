package net.github.gradientgmm

import breeze.linalg.{DenseVector => BDV}

class WeightsWrapper(var weights: Array[Double]) extends Serializable{

  require(checkPositivity(weights), "some weights are negative or equal to zero")
  require(isInSimplex(weights),s"new weights don't sum 1: ${weights.mkString(",")}")

  var simplexErrorTol = 1e-8
  var momentum: Option[BDV[Double]] = None
  var adamInfo: Option[BDV[Double]] = None
  var length = weights.length

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

  private[gradientgmm] def updateMomentum(x: BDV[Double]): Unit = {
    momentum = Option(x)
  }

  private[gradientgmm] def updateAdamInfo(x: BDV[Double]): Unit = {
    adamInfo = Option(x)
  }

  private[gradientgmm] def initializeMomentum: Unit = {
    momentum = Option(BDV.zeros[Double](weights.length))
  }

  private[gradientgmm] def initializeAdamInfo: Unit = {
     adamInfo = Option(BDV.zeros[Double](weights.length))
  }

}