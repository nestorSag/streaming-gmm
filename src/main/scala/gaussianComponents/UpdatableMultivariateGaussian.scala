package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMVG}

class UpdatableMultivariateGaussian private(
  _s: Double, 
  _mu: BDV[Double], //there is a bug when using the same attribute names for the subclass and superclass
  _sigma: BDM[Double]) extends GConcaveMultivariateGaussian(_s,_mu,_sigma) with GradientDescentUtils {

  def update(newParamsMat: BDM[Double]): Unit = {

    this.s = newParamsMat(d,d)
    this.mu = newParamsMat(0 to d-1,d)/s
    this.sigma = newParamsMat(0 to d-1,0 to d-1) - (mu)*(mu).t*s

    //logger.debug(s"mu: ${mu}")

    var (rootSigmaInv_,u_) = calculateCovarianceConstants


    this.rootSigmaInv = rootSigmaInv_
    this.u = u_
  }

}


object UpdatableMultivariateGaussian {

  def apply(s: Double, mu: BDV[Double], sigma: BDM[Double]): UpdatableMultivariateGaussian = {
    new UpdatableMultivariateGaussian(s,mu,sigma)
  }

  def apply(mu: BDV[Double], sigma: BDM[Double]): UpdatableMultivariateGaussian = {
    UpdatableMultivariateGaussian(1.0,mu,sigma)
  }

  def apply(mu: SV, sigma: SDM): UpdatableMultivariateGaussian = {
    UpdatableMultivariateGaussian(1.0,mu,sigma)
  }

  def apply(s: Double, mu: SV, sigma: SDM) = {
    new UpdatableMultivariateGaussian(
      s, 
      new BDV[Double](mu.toArray), 
      new BDM[Double](sigma.numRows,sigma.numCols,sigma.toArray))
  }

  def apply(s: BDM[Double]) = {

    val newS = s(s.rows-1,s.cols-1)
    val newMu = s(0 to s.rows-2,s.cols-1)/newS

    new UpdatableMultivariateGaussian(
      newS, 
      newMu,
      s(0 to s.rows-2,0 to s.cols-2) - newMu*newMu.t*newS)
  }

  def apply(g: SMVG) = {

    new UpdatableMultivariateGaussian(
      1.0, 
      new BDV(g.mu.toArray),
      new BDM(g.mu.size,g.mu.size,g.sigma.toArray))
  }
}