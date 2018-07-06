package edu.github.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

class MultivariateGaussian(
  private[gradientgmm] var mu: BDV[Double],
  private[gradientgmm] var sigma: BDM[Double]) extends Serializable{

  var (rootSigmaInv: BDM[Double], u: Double) = calculateCovarianceConstants
  val d = mu.length
  private val EPS = Utils.EPS

  require(sigma.cols == sigma.rows, "Covariance matrix must be square")
  require(mu.length == sigma.cols, "Mean vector length must match covariance matrix size")

  def getMu: BDV[Double] = this.mu

  def getSigma: BDM[Double] = this.sigma

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

  def logDetSigma(): Double = {
    -2.0*u - mu.size * math.log(2.0 * math.Pi)
  }

  def detSigma(): Double = {
    math.exp(logDetSigma)
  }


  private[gradientgmm] def calculateCovarianceConstants: (BDM[Double], Double) = {
    val eigSym.EigSym(d, u) = eigSym(sigma) // sigma = u * diag(d) * u.t

    // For numerical stability, values are considered to be non-zero only if they exceed tol.
    // This prevents any inverted value from exceeding (eps * n * max(d))^-1

    val tol = EPS * max(d) * d.length

    try {
      // log(pseudo-determinant) is sum of the logs of all non-zero singular values

      val logPseudoDetSigma = d.activeValuesIterator.filter(_ > tol).map(math.log).sum

      //val logPseudoDetSigma = d.map(math.log).sum


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