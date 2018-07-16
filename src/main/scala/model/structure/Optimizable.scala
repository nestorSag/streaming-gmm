package com.github.gradientgmm.optim.algorithms

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Contains the basic functionality for an object to be modified by {{{Optimizer}}}

  */
trait Optimizable extends Serializable {

/**
  * optimizer object

  */
  protected var optimizer: Optimizer

  def setOptimizer(optim: Optimizer): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: Optimizer = optimizer

/**
  * Perform a gradient-based optimization step
  * @param data Data to fit the model
  */
  def step(data: RDD[SV]): Unit

  /**
  * Linear Algebra operations necessary for computing updates for the parameters
    
  * This is to avoid duplicating code for Gaussian and Weights updates in the optimization
  * algorithms' classes
 
  */
  protected implicit val vectorOps = new ParameterOperations[BDV[Double]] {
    def sum(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x + y}
    def sumScalar(x: BDV[Double], z: Double): BDV[Double] = {x + z}
    def rescale(x: BDV[Double], z: Double): BDV[Double] = {x*z}
    def sub(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x - y}

    def ewProd(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x *:* y}
    def ewDiv(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x /:/ y}
    def ewSqrt(x:BDV[Double]): BDV[Double] = {sqrt(x)}
  }

  protected implicit val matrixOps = new ParameterOperations[BDM[Double]] {
    def sum(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x + y}
    def sumScalar(x: BDM[Double], z: Double): BDM[Double] = {x + z}
    def rescale(x: BDM[Double], z: Double): BDM[Double] = {x*z}
    def sub(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x - y}

    def ewProd(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x *:* y}
    def ewDiv(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x /:/ y}
    def ewSqrt(x:BDM[Double]): BDM[Double] = {sqrt(x)}
  }

}

/**
  * Contains common mathematical operations that can be performed in both matrices and vectors.
  * Its purpose is avoid duplicating code in the optimization algorithms' classes
  */
trait ParameterOperations[A] extends Serializable{

  def sum(x: A, y: A): A
  def sumScalar(x:A,z:Double): A
  def rescale(x: A, d: Double): A
  def sub(x:A, y:A): A

  def ewProd(x:A,y:A): A
  def ewDiv(x:A,y:A): A
  def ewSqrt(x:A): A
}