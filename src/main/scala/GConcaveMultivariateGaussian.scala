import breeze.linalg.{diag, eigSym, max, DenseMatrix => DBM, DenseVector => DBV, Vector => BV}

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian


/**
 * :: DeveloperApi ::
 * This class provides basic functionality for a Multivariate Gaussian (Normal) Distribution. In
 * the event that the covariance matrix is singular, the density will be computed in a
 * reduced dimensional subspace under which the distribution is supported.
 * (see <a href="http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Degenerate_case">
 * Degenerate case in Multivariate normal distribution (Wikipedia)</a>)
 *
 * @param mu The mean vector of the distribution
 * @param sigma The covariance matrix of the distribution
 */


class GConcaveMultivariateGaussian (
	val mu: Vector, 
	val sigma: Matrix, 
	val s: Double) extends MultivariateGaussian(mu,sigma) = {

	require(s > 0, s"s must be positive; got ${s}")

	this(mu: Vector, sigma: Matrix): this.type = {
		this(mu,sigma,1)
	}
	
	def gConcavePdf(x: Vector): Double = {
		pdf(x) * exp(0.5*(1-1/s)) / math.sqrt(s)
	}

	def gConcaveLogPdf(x: Vector): Double = {
		math.log(gConcavePdf(x))
	}

}