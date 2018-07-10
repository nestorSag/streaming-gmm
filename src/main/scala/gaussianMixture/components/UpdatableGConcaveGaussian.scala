package com.github.nestorsag.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMVG}

/**
  * Gaussian distribution that implements an updating routine based on its g-concave reformulation 
  * and contains gradient ascent utilities. It can only be initialized bi its companion object 

  * @param _s Positive scalar
  * @param _mu Mean vector
  * @param _sigma: Covariance matrix
 
  */
class UpdatableGConcaveGaussian private(
  _s: Double, 
  _mu: BDV[Double], //there is a bug when using the same attribute names for the subclass and superclass
  _sigma: BDM[Double]) extends GConcaveGaussian(_s,_mu,_sigma) with MatrixOptimUtils {

/**
  * accelerated gradient descent utilities. See [[AcceleratedGradientUtils]]
 
  */
  val optimUtils = new MatrixGradientUtils(d+1)

/**
  * Given a matrix, updates the distribution's parameters according to the 
  * augmented parameter matrix' block structure

  * @param newParamMat Matrix with new parameters
 
  */
  def update(newParamsMat: BDM[Double]): Unit = {

    this.s = newParamsMat(d,d)
    this.mu = newParamsMat(0 to d-1,d)/s
    this.sigma = newParamsMat(0 to d-1,0 to d-1) - (mu)*(mu).t*s
    this.paramMat = computeParamMat
  
    var (rootSigmaInv_,u_) = calculateCovarianceConstants


    this.rootSigmaInv = rootSigmaInv_
    this.u = u_
  }

}


object UpdatableGConcaveGaussian {

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance.


  * @param s Positive scalar
  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */
  def apply(s: Double, mu: BDV[Double], sigma: BDM[Double]): UpdatableGConcaveGaussian = {
    new UpdatableGConcaveGaussian(s,mu,sigma)
  }

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance with default parameter {{{s = 1}}}.


  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */
  def apply(mu: BDV[Double], sigma: BDM[Double]): UpdatableGConcaveGaussian = {
    UpdatableGConcaveGaussian(1.0,mu,sigma)
  }

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance with default parameter {{{s = 1}}}.


  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */
  def apply(mu: SV, sigma: SDM): UpdatableGConcaveGaussian = {
    UpdatableGConcaveGaussian(1.0,mu,sigma)
  }

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance.


  * @param s Positive scalar
  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */
  def apply(s: Double, mu: SV, sigma: SDM) = {
    new UpdatableGConcaveGaussian(
      s, 
      new BDV[Double](mu.toArray), 
      new BDM[Double](sigma.numRows,sigma.numCols,sigma.toArray))
  }

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance from a block matrix.


  * @param mat Matrix with augmented parameter matrix structure
 
  */
  def apply(mat: BDM[Double]) = {

    val newS = mat(mat.rows-1,mat.cols-1)
    val newMu = mat(0 to mat.rows-2,mat.cols-1)/newS

    new UpdatableGConcaveGaussian(
      newS, 
      newMu,
      mat(0 to mat.rows-2,0 to mat.cols-2) - newMu*newMu.t*newS)
  }

/**
  * Creates an {{{UpdatableGConcaveGaussian}}} instance from an [[[org.apache.spark.ml.stat.distribution.MultivariateGaussian]]]
  * instance

  * @param g Spark's Multivariate Gaussian instance
 
  */
  def apply(g: SMVG) = {

    new UpdatableGConcaveGaussian(
      1.0, 
      new BDV(g.mu.toArray),
      new BDM(g.mu.size,g.mu.size,g.sigma.toArray))
  }
}