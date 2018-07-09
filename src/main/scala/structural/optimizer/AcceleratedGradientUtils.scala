package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

abstract class AcceleratedGradientUtils[T <: Any](val zeroGenerator: Int => T, val d: Int) extends Serializable{

  var momentum: Option[T] = None
  var adamInfo: Option[T] = None

  def updateMomentum(x: T): Unit = {
    momentum = Option(x)
  }

  def updateAdamInfo(x: T): Unit = {
    adamInfo = Option(x)
  }

  def initializeMomentum: Unit = {
    momentum = Option(zeroGenerator(d))
  }

  def initializeAdamInfo: Unit = {
    adamInfo = Option(zeroGenerator(d))
  }

  def lighten: this.type = {
    momentum = None
    adamInfo = None
    this
  }
}

class VectorGradientUtils(
  d: Int) extends AcceleratedGradientUtils[BDV[Double]]((n:Int) => BDV.zeros[Double](n),d)


trait VectorOptimUtils extends Serializable{
  val optimUtils: VectorGradientUtils
}

class MatrixGradientUtils(
  d: Int) extends AcceleratedGradientUtils[BDM[Double]]((n:Int) => BDM.zeros[Double](n,n),d)


trait MatrixOptimUtils extends Serializable{
  val optimUtils: MatrixGradientUtils
}

