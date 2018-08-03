package com.github.gradientgmm.components

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

/**
  * Multivariate Gaussian Distribution reformulation that implies a g-concave loss function in 
  * [[https://arxiv.org/abs/1706.03267 An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization'']]
  *
  * For an arbitrary Gaussian distribution, its g-concave reformulation have zero mean and an
  * augmented covariance matrix which is a function of the original mean, covariance matrix
  * and an additional positive scalar s. Original data points x are mapped to 
  * y = [x 1] to be evaluated under the new distribution. When s = 1, the density
  * of the original distirbution and the reformulation are equal for all points.

  * @param s Positive scalar
  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */
class GConcaveGaussian(
	var s: Double,
	_mu: BDV[Double],
  _sigma: BDM[Double]) extends MultivariateGaussian(_mu,_sigma){

require(s > 0, s"s must be positive; got ${s}")

  def getS: Double = this.s

/**
  * Returns the g-concave reformulation's density evaluated on x
 
  */
  def gConcavePdf(x: BDV[Double]): Double = {
    pdf(x.slice(0,d)) * math.exp(0.5*(1-1/s)) / math.sqrt(s)
  }

/**
  * Augmented parameter block matrix [A B; C D]. The blocks are:

  * A = sigma + s * mu * mu.t

  * B = s * mu

  * C = s * mu.t

  * D = s
 
  */
  var paramMat: BDM[Double] = computeParamMat


/**
  * Augmented parameter block matrix inverse. Its blocks are:

  * A = sigmaInv

  * B = sigmaInv * mu

  * C = mu.t * sigmaInv

  * D = 1/s + mu.t * sigmaInv * mu
 
  */
  def invParamMat: BDM[Double] = {
    // build S inv matrix

    val x = this.rootSigmaInv.t*this.rootSigmaInv*mu
    val lastRow = new BDV[Double](x.toArray ++ Array[Double](-1/s - mu.t*x))
    
    BDM.vertcat(BDM.horzcat(this.rootSigmaInv.t*this.rootSigmaInv,-x.asDenseMatrix.t),-lastRow.asDenseMatrix)

  }

/**
  * Compute augmented parameter matrix
 
  */
  def computeParamMat: BDM[Double] = {
    // build S matrix
    val lastRow = new BDV[Double](mu.toArray ++ Array[Double](1))

    BDM.vertcat(BDM.horzcat(sigma + mu*mu.t*s,mu.asDenseMatrix.t*s),lastRow.asDenseMatrix*s)

  }


}