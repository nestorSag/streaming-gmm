
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}

import scala.collection.mutable.IndexedSeq

import breeze.linalg.{diag, DenseMatrix => BreezeMatrix, DenseVector => BDV, Vector => BV}

import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{BLAS, DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

/**
 * This class performs expectation maximization for multivariate Gaussian
 * Mixture Models (GMMs).  A GMM represents a composite distribution of
 * independent Gaussian distributions with associated "mixing" weights
 * specifying each's contribution to the composite.
 *
 * Given a set of sample points, this class will maximize the log-likelihood
 * for a mixture of k Gaussians, iterating until the log-likelihood changes by
 * less than convergenceTol, or until it has reached the max number of iterations.
 * While this process is generally guaranteed to converge, it is not guaranteed
 * to find a global optimum.
 *
 * @param k Number of independent Gaussians in the mixture model.
 * @param convergenceTol Maximum change in log-likelihood at which convergence
 *                       is considered to have occurred.
 * @param maxEMIterations Maximum number of EM iterations allowed.
 *
 * @note This algorithm is limited in its number of features since it requires storing a covariance
 * matrix which has size quadratic in the number of features. Even when the number of features does
 * not exceed this limit, this algorithm may perform poorly on high-dimensional data.
 * This is due to high-dimensional data (a) making it difficult to cluster at all (based
 * on statistical/theoretical arguments) and (b) numerical issues with Gaussian distributions.
 */
class SGDGaussianMixture private (
    private var k: Int,
    private var convergenceTol: Double,
    private var maxEMIterations: Int,
    private var seed: Long,
    private var learningRate: Double) extends GaussianMixture(k,convergenceTol,maxEMIterations,seed) {

  def getLearningRate: Double = learningRate

  def setLearningRate(alpha: Double): this.type = {
    require(alpha > 0,
      s"learning rate must be positive; got ${alpha}")
    this.learningRate = alpha
    this
  }
  
  def step(data: RDD[Vector], maxiter: Double = 1): GaussianMixtureModel = {
    val sc = data.sparkContext

    // we will operate on the data as breeze data
    val breezeData = data.map{x => x.toBreeze()}.cache()

    // Get length of the input vectors
    val d = breezeData.first().length
    require(d < GaussianMixture.MAX_NUM_FEATURES, s"GaussianMixture cannot handle more " +
      s"than ${GaussianMixture.MAX_NUM_FEATURES} features because the size of the covariance" +
      s" matrix is quadratic in the number of features.")

    val shouldDistributeGaussians = GaussianMixture.shouldDistributeGaussians(k, d)

    // Determine initial weights and corresponding Gaussians.
    // If the user supplied an initial GMM, we use those values, otherwise
    // we start with uniform weights, a random mean from the data, and
    // diagonal covariance matrices using component variances
    // derived from the samples
    val (weights, gaussians) = initialModel match {
      case Some(gmm) => (gmm.weights, gmm.gaussians)

      case None =>
        val samples = breezeData.takeSample(withReplacement = true, k * nSamples, seed)
        (Array.fill(k)(1.0 / k), Array.tabulate(k) { i =>
          val slice = samples.view(i * nSamples, (i + 1) * nSamples)
          new GConcaveMultivariateGaussian(vectorMean(slice), initCovariance(slice))
        })
    }

    var llh = Double.MinValue // current log-likelihood
    var llhp = 0.0            // previous log-likelihood

    var iter = 0
    while (iter < maxiter && math.abs(llh-llhp) > convergenceTol) {
      // create and broadcast curried cluster contribution function
      val compute = sc.broadcast(ExpectationSum.add(weights, gaussians)_)

      // aggregate the cluster contribution for all sample points
      val sums = breezeData.treeAggregate(ExpectationSum.zero(k, d))(compute.value, _ += _)

      val n = sums.weights.sum // number of data points 

      val tuples =
          Seq.tabulate(k)(i => (sums.means(i), sums.sigmas(i), sums.weights(i), weights(i), gaussians(i)), i)

      if (shouldDistributeGaussians) {
        val numPartitions = math.min(k, 1024)

        val gaussians = sc.parallelize(tuples, numPartitions).map { case (mean, sigma, weight, currentWeight, currentDist) =>
          gradientAscent(currentWeight, currentDist, mean, sigma, weight, n, i)
        }.collect().unzip

        //Array.copy(ws, 0, weights, 0, ws.length)
        //Array.copy(gs, 0, gaussians, 0, gs.length)

      } else {

        val gaussians = tuples.map{ 
          case (mean, sigma, weight, currentWeight, currentDist) => 
          gradientAscent(currentWeight, currentDist, mean, sigma, weight, n)}
      }

      llhp = llh // current becomes previous
      llh = sums.logLikelihood // this is the freshly computed log-likelihood
      iter += 1
      compute.destroy(blocking = false)
    }

    new GaussianMixtureModel(weights, gaussians)
  }

  private def weightsGradientAscent(weights: Array[Double], posteriorProbs: Array[Double], n: Double): Array[Double] = {

    val cumWeights = 0
    for(i <- 0 to k-2){
      weights(i) *= math.exp(learningRate*(posteriorProbs(i)/n-weights(i)))
      cumWeights += weights(i)
    }
    weights(k-1) = 1 - cumWeights
    weights
  }

  private def gradientAscent(
      currentWeight: Double,
      currentDist: GConcaveMultivariateGaussian,
      mean: BDV[Double],
      sigma: BreezeMatrix[Double],
      weight: Double,
      n: Double,
      cluster: Int): GConcaveMultivariateGaussian = {

    val avgRate = learningRate/(2*n)
    val shrinkage = (1 - avgRate*weight)

    val newMu = shrinkage * currentDist.mu + avgRate*mean
    val newSigma = shrinkage * currentDist.sigma + avgRate*sigma
   
    val weight = cluster match compare k-1 {
      case -1 => currentWeight
    }
    val newGaussian = new GConcaveMultivariateGaussian(mu, sigma / weight)
    newGaussian
  }













//   private object GConcaveExpectationSum {
//     def zero(k: Int, d: Int): GConcaveExpectationSum = {
//       new GConcaveExpectationSum(0.0, Array.fill(k)(0.0),
//         Array.fill(k)(BDV.zeros(d)), Array.fill(k)(BreezeMatrix.zeros(d, d)))
//     }

//     // compute cluster contributions for each input point
//     // (U, T) => U for aggregation
//     def add(
//         weights: Array[Double],
//         dists: Array[GConcaveMultivariateGaussian])
//         (sums: GConcaveExpectationSum, x: BV[Double]): GConcaveExpectationSum = {
//       val p = weights.zip(dists).map {
//         case (weight, dist) => MLUtils.EPSILON + weight * dist.gConcavePdf(x) // <--only real change
//       }
//       val pSum = p.sum
//       sums.logLikelihood += math.log(pSum)
//       var i = 0
//       while (i < sums.k) {
//         p(i) /= pSum
//         sums.weights(i) += p(i)
//         sums.means(i) += x * p(i)
//         BLAS.syr(p(i), Vectors.fromBreeze(x),
//           Matrices.fromBreeze(sums.sigmas(i)).asInstanceOf[DenseMatrix])
//         i = i + 1
//       }
//       sums
//     }
// }

// // Aggregation class for partial expectation results
// private class GConcaveExpectationSum(
//     var logLikelihood: Double,
//     val weights: Array[Double],
//     val means: Array[BDV[Double]],
//     val sigmas: Array[BreezeMatrix[Double]]) extends Serializable {

//     val k = weights.length

//     def +=(x: GConcaveExpectationSum): GConcaveExpectationSum = {
//       var i = 0
//       while (i < k) {
//         weights(i) += x.weights(i)
//         means(i) += x.means(i)
//         sigmas(i) += x.sigmas(i)
//         i = i + 1
//       }
//       logLikelihood += x.logLikelihood
//       this
//     }

// }