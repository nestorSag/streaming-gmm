package com.github.gradientgmm.optim

import breeze.linalg.{diag, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.{sqrt,abs}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD

import org.apache.spark.streaming.api.java.JavaDStream
import org.apache.spark.streaming.dstream.DStream

/**
  * Contains basic functionality for an object that can be modified by Optimizer

  */
trait Optimizable extends Serializable {

/**
  * Current loss value
  */
  protected var lossValue: Double = Double.MinValue

/**
  * convergence flag for local optimisation
  */
  protected var converged = false


/**
  * Optional regularization term
  */
  protected var regularizer: Option[Regularizer] = None


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

/**
  * random seed for mini-batch sampling

  */
  protected var seed: Long = 0

/**
  * this prevents the seed from repeating every time step() is called
  * which would cause the same samples being taken

  */
  protected implicit var globalIterCounter: Long = 0

  def setOptim(optim: Optimizer): this.type = {
    this.optim = optim
    this
  }

  def getOptim: Optimizer = optim

/**
  * Perform a gradient-based optimization step
  * @param data Data to fit the model
  */
  def step(data: RDD[SV]): this.type

/**
  * Perform a gradient-based optimization step
  * @param data Data to fit the model
  */
  def step(data: JavaRDD[SV]): Unit = {
    step(data.rdd)
  }

//functionality for streaming data

/**
  * Update model parameters using streaming data
  * @param data Streaming data
 
  */
  def step(data: DStream[SV]): this.type = {
    data.foreachRDD { (rdd, time) =>
      step(rdd)
    }
    this
  }

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

    def ewAbs(x:BDV[Double]): BDV[Double] = {abs(x)}
    def ewProd(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x *:* y}
    def ewDiv(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x /:/ y}
    def ewSqrt(x:BDV[Double]): BDV[Double] = {sqrt(x)}
  }

  protected implicit val matrixOps = new ParameterOperations[BDM[Double]] {
    def sum(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x + y}
    def sumScalar(x: BDM[Double], z: Double): BDM[Double] = {x + z}
    def rescale(x: BDM[Double], z: Double): BDM[Double] = {x*z}
    def sub(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x - y}
    
    def ewAbs(x:BDM[Double]): BDM[Double] = {abs(x)}
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

  def setBatchSize(n: Option[Int]): this.type = {
    if(n.isDefined){
      require(n.get>0,"n must be a positive integer")
      batchSize = n
    }else{
      batchSize = None
    }

    this
  }

  def setSeed(s: Long): this.type = {
    seed = s
    this
  }

  def getSeed: Long = seed

  def setRegularizer(r: Regularizer): this.type = {
    regularizer = Option(r)
    this
  }

  def setRegularizer(r: Option[Regularizer]): this.type = {
    regularizer = r
    this
  }

  def getLoss = lossValue


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

  def ewAbs(x: A): A
  def ewProd(x:A,y:A): A
  def ewDiv(x:A,y:A): A
  def ewSqrt(x:A): A
}