package com.github.gradientgmm.components

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

/**
  * Multivariate Gaussian distribution class
  *
  * It is based on [[org.apache.spark.ml.stat.distribution.MultivariateGaussian]] and it implements 
  * much of the same functionality, but more of its internal objects, such as the covariance matrix determinant
  * and the covariance matrix inverse, are public; many of its parameters are mutable as well. 

  * @param mu Mean vector
  * @param sigma: Covariance matrix
 
  */

class MultivariateGaussian(
  private[gradientgmm] var mu: BDV[Double],
  private[gradientgmm] var sigma: BDM[Double]) extends Serializable{

/**
  * square root of the covariance matrix inverse, and the density's constant term
 
  */
  var (rootSigmaInv: BDM[Double], u: Double) = calculateCovarianceConstants

/**
* data dimensionality

*/
  val d = mu.length

/**
* machine's epsilon

*/
  private val EPS = Utils.EPS

  require(sigma.cols == sigma.rows, "Covariance matrix must be square")
  require(mu.length == sigma.cols, "Mean vector length must match covariance matrix size")

  def getMu: BDV[Double] = this.mu

  def getSigma: BDM[Double] = this.sigma

/**
* Returns the distribution's density function evaluated on x

*/
  def pdf(x: SV): Double = {
    pdf(new BDV(x.toArray))
  }

/**
* Returns the distribution's log-density function evaluated on x

*/
  def logpdf(x: SV): Double = {
    logpdf(new BDV(x.toArray))
  }

/**
* Returns the distribution's density function evaluated on x

*/
  def pdf(x: BV[Double]): Double = {
    math.exp(logpdf(x))
  }

/**
* Returns the distribution's log-density function evaluated on x

*/
  def logpdf(x: BV[Double]): Double = {
    val delta = x - mu
    val v = rootSigmaInv * delta
    u + v.t * v * -0.5
  }

/**
* Returns the covariance matrix' log-determinant

*/
  def logDetSigma: Double = {
    -2.0*u - mu.size * math.log(2.0 * math.Pi)
  }

/**
* Returns the covariance matrix' determinant

*/
  def detSigma: Double = {
    math.exp(logDetSigma)
  }

/**
* THe following was taken from the comments in the source code of
* [[[[org.apache.spark.ml.stat.distribution.MultivariateGaussian]]]]:


*/

  /**
   * Calculate distribution dependent components used for the density function:
   *    pdf(x) = (2*pi)^(-k/2)^ * det(sigma)^(-1/2)^ * exp((-1/2) * (x-mu).t * inv(sigma) * (x-mu))
   * where k is length of the mean vector.
   *
   * We here compute distribution-fixed parts
   *  log((2*pi)^(-k/2)^ * det(sigma)^(-1/2)^)
   * and
   *  D^(-1/2)^ * U, where sigma = U * D * U.t
   *
   * Both the determinant and the inverse can be computed from the singular value decomposition
   * of sigma.  Noting that covariance matrices are always symmetric and positive semi-definite,
   * we can use the eigendecomposition. We also do not compute the inverse directly; noting
   * that
   *
   *    sigma = U * D * U.t
   *    inv(Sigma) = U * inv(D) * U.t
   *               = (D^{-1/2}^ * U.t).t * (D^{-1/2}^ * U.t)
   *
   * and thus
   *
   *    -0.5 * (x-mu).t * inv(Sigma) * (x-mu) = -0.5 * norm(D^{-1/2}^ * U.t  * (x-mu))^2^
   *
   * To guard against singular covariance matrices, this method computes both the
   * pseudo-determinant and the pseudo-inverse (Moore-Penrose).  Singular values are considered
   * to be non-zero only if they exceed a tolerance based on machine precision, matrix size, and
   * relation to the maximum singular value (same tolerance used by, e.g., Octave).
   */

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