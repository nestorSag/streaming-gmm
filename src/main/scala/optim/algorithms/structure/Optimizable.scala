package com.github.gradientgmm.optim.algorithms

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Contains the basic functionality for an object to be modified by Optimizer

  */
trait Optimizable extends Serializable {

/**
  * Minibatch size for each iteration in the ascent procedure. If None, it performs
  * full-batch optimization
  */
  protected var batchSize: Option[Int] = None

/**
  * Error tolerance in log-likelihood for the stopping criteria
  */
  protected var convergenceTol: Double = 1e-6

/**
  * Maximum number of iterations allowed
  */
  protected var maxIter: Int = 100

/**
  * optimizer object

  */
  protected var optim: Optimizer

  def setOptim(optim: Optimizer): this.type = {
    this.optim = optim
    this
  }

  def getOptim: Optimizer = optim

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


  def getConvergenceTol: Double = convergenceTol

  def setConvergenceTol(x: Double): this.type = {
    require(x>0,"convergenceTol must be positive")
    convergenceTol = x
    this
  }


  def setMaxIter(m: Int): this.type = {
    require(m > 0 ,s"maxIter needs to be a positive integer; got ${m}")
    maxIter = m
    this
  }

  def getMaxIter: Int = {
    maxIter
  }

  def getBatchSize: Option[Int] = batchSize

  def setBatchSize(n: Int): this.type = {
    require(n>0,"n must be a positive integer")
    batchSize = Option(n)
    this
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