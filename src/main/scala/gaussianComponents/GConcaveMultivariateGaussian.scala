package edu.github.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

class GConcaveMultivariateGaussian(
	var s: Double,
	_mu: BDV[Double],
  _sigma: BDM[Double]) extends MultivariateGaussian(_mu,_sigma){

require(s > 0, s"s must be positive; got ${s}")

  def getS: Double = this.s
  
  def gConcavePdf(x: BDV[Double]): Double = {
    pdf(x.slice(0,d)) * math.exp(0.5*(1-1/s)) / math.sqrt(s)
  }

  def paramMat: BDM[Double] = {
    // build S matrix
    val lastRow = new BDV[Double](mu.toArray ++ Array[Double](1))

    BDM.vertcat(BDM.horzcat(sigma + mu*mu.t*s,mu.asDenseMatrix.t*s),lastRow.asDenseMatrix*s)

  }

  def invParamMat: BDM[Double] = {
    // build S inv matrix

    val x = this.rootSigmaInv.t*this.rootSigmaInv*mu
    val lastRow = new BDV[Double](x.toArray ++ Array[Double](-1/s - mu.t*x))
    
    BDM.vertcat(BDM.horzcat(this.rootSigmaInv.t*this.rootSigmaInv,-x.asDenseMatrix.t),-lastRow.asDenseMatrix)

  }
}