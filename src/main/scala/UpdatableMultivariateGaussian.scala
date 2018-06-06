package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian

class UpdatableMultivariateGaussian private(
	private var s: Double,
  private var mu: BDV[Double],
  private var sigma: BDM[Double]) extends Serializable {

  //var rootSigmaInv: BDM[Double]

  //var u: Double

  val d = mu.length

  private[streamingGmm] var momentum: Option[BDM[Double]] = None

  private lazy val eps = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  require(sigma.cols == sigma.rows, "Covariance matrix must be square")
  require(mu.length == sigma.cols, "Mean vector length must match covariance matrix size")
  require(s > 0, s"s must be positive; got ${s}")

  def getS: Double = this.s

  def getMu: BDV[Double] = this.mu

  def getSigma: BDM[Double] = this.sigma

  var (rootSigmaInv: BDM[Double], u: Double) = calculateCovarianceConstants

  def pdf(x: SV): Double = {
    pdf(new BDV(x.toArray))
  }

  def logpdf(x: SV): Double = {
    logpdf(new BDV(x.toArray))
  }

  def pdf(x: BV[Double]): Double = {
    math.exp(logpdf(x))
  }

  def logpdf(x: BV[Double]): Double = {
    val delta = x - mu
    val v = rootSigmaInv * delta
    u + v.t * v * -0.5
  }

  def detSigma(): Double = {
  	math.exp(-0.5*u - mu.size * math.log(2.0 * math.Pi))
  }
  
  def gConcavePdf(x: BV[Double]): Double = {
    pdf(x(0,d)) * math.exp(0.5*(1-1/s)) / math.sqrt(s)
  }

  def gConcaveLogPdf(x: BV[Double]): Double = {
    math.log(gConcavePdf(x))
  }

  def paramMat: BDM[Double] = {
    // build S matrix
    val lastRow = new BDV[Double](mu.toArray ++ Array[Double](1))

    BDM.vertcat(BDM.horzcat(sigma/s + mu*mu.t,mu.asDenseMatrix.t),lastRow.asDenseMatrix) * s

  }

  private def update(newParamsMat: BDM[Double]): Unit = {

    this.s = newParamsMat(d-1,d-1)
    this.mu = newParamsMat(0 to d-2,d-1)/this.s
    this.sigma = newParamsMat(0 to d-2,0 to d-1) - this.mu*this.mu.t*s

    var (rootSigmaInv_,u_) = calculateCovarianceConstants

    this.rootSigmaInv = rootSigmaInv_

    this.u = u_
  }

  def step(cov: BDM[Double], optimizer: GMMGradientAscent, n:Double): this.type = {
    this.update(this.paramMat + optimizer.direction(this,cov)*optimizer.learningRate/n)
    this
  }

  private def calculateCovarianceConstants: (BDM[Double], Double) = {
    val eigSym.EigSym(d, u) = eigSym(sigma) // sigma = u * diag(d) * u.t

    // For numerical stability, values are considered to be non-zero only if they exceed tol.
    // This prevents any inverted value from exceeding (eps * n * max(d))^-1

    val tol = eps * max(d) * d.length

    try {
      // log(pseudo-determinant) is sum of the logs of all non-zero singular values

      //val logPseudoDetSigma = d.activeValuesIterator.filter(_ > tol).map(math.log).sum

      val logPseudoDetSigma = d.map(math.log).sum


      // calculate the root-pseudo-inverse of the diagonal matrix of singular values
      // by inverting the square root of all non-zero values
      val pinvS = diag(new BDV(d.map(v => if (v > tol) math.sqrt(1.0 / v) else 0.0).toArray))

      (pinvS * u.t, -0.5 * (mu.size * math.log(2.0 * math.Pi) + logPseudoDetSigma))
    } catch {
      case uex: UnsupportedOperationException =>
        throw new IllegalArgumentException("Covariance matrix has no non-zero singular values")
    }
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
}