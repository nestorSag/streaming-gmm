package com.github.nestorsag.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Distributed aggregator of relevant statistics
  *
  * In each worker it computes and aggregates the current batch log-likelihood,
  * the regularization values for the current parameters and the 
  * gaussianGradients for each data point. The class structure is based heavily on
  * Spark Clustering's {{{ExpectationSum}}} class. 
  * See [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/GaussianMixture.scala]]

  * @param qLogLikelihood aggregate log-likelihood
  * @param weightsGradient: aggregate posterior responsability for each component. See ''Pattern Recognition
  * And Machine Learning. Bishop, Chis.'', page 432
  * @param gaussianGradients Aggregate point-wise gaussianGradients for each component
 
  */
class GradientAggregator(
  var qLoglikelihood: Double,
  val weightsGradient: BDV[Double],
  val gaussianGradients: Array[BDM[Double]],
  var counter: Int) extends Serializable{

/**
  * Number of components in the model
  */
  val k = gaussianGradients.length

/**
  * Adder for different {{{GradientAggregator}}}
  *
  * Used for further aggregation between each worker's object
 
  */
  def +=(x: GradientAggregator): GradientAggregator = {
    var i = 0
    while (i < k) {
      gaussianGradients(i) += x.gaussianGradients(i)
      i += 1
    }
    weightsGradient += x.weightsGradient
    qLoglikelihood += x.qLoglikelihood
    counter += x.counter
    this
  }

}

object GradientAggregator {

/**
  * {{{GradientAggregator}}} initializer
  *
  * Initializes an instance with initial statistics set as zero
  * @param k Number of components in the model
  * @param d Dimensionality of the data
 
  */
  def init(k: Int, d: Int): GradientAggregator = {
    new GradientAggregator(
      0.0,
      BDV.zeros[Double](k),
      Array.fill(k)(BDM.zeros[Double](d+1, d+1)),
      0)
  }

/**
  * Adder for individual points
  *
  * Used for reducing individual data points and aggregating the ir statistics
  * @param weights Current weights vector
  * @param dists Current model components
  * @param optim Optimization algorithm
  * @param Number of points in the current batch
  * @return Instance with updated statistics
 
  */
  def add(
      weights: Array[Double],
      dists: Array[UpdatableGaussianMixtureComponent],
      optim: GMMOptimizer,
      n: Double)
      (agg: GradientAggregator, y: BDV[Double]): GradientAggregator = {

    agg.counter += 1

    val q = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.gConcavePdf(y) // <--q-logLikelihood
    }
    val qSum = q.sum

    agg.qLoglikelihood += math.log(qSum) / n

    // update aggregated weight gradient
    val posteriors = Utils.toBDV(q) / qSum
    agg.weightsGradient += optim.weightsGradient(posteriors,Utils.toBDV(weights)) / n

    // update gaussian parameters' gradients and log-likelihood
    var i = 0
    val outer = y*y.t
    while (i < agg.k) {
      q(i) /= qSum 

      agg.gaussianGradients(i) += optim.gaussianGradient( dists(i), outer , q(i)) / n

      agg.qLoglikelihood += optim.evaluateRegularizationTerm(dists(i),weights(i)) / (n*n)

      i = i + 1
    }
    agg
  }
}