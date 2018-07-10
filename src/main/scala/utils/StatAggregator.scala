package com.github.nestorsag.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

/**
  * Distributed aggregator of relevant statistics
  *
  * In each worker it computes and aggregates the current batch log-likelihood,
  * the regularization values for the current parameters and the 
  * gradients for each data point. The class structure is based heavily on
  * Spark Clustering's {{{ExpectationSum}}} class. 
  * See [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/GaussianMixture.scala]]

  * @param qLogLikelihood aggregate log-likelihood
  * @param posteriors: aggregate posterior responsability for each component. See ''Pattern Recognition
  * And Machine Learning. Bishop, Chis.'', page 432
  * @param gradients Aggregate point-wise gradients for each component
 
  */
class StatAggregator(
  var qLoglikelihood: Double,
  val posteriors: Array[Double],
  val gradients: Array[BDM[Double]]) extends Serializable{

/**
  * Number of components in the model
  */
  val k = gradients.length

/**
  * Adder for different {{{StatAggregator}}}
  *
  * Used for further aggregation between each worker's object
 
  */
  def +=(x: StatAggregator): StatAggregator = {
    var i = 0
    while (i < k) {
      gradients(i) += x.gradients(i)
      posteriors(i) += x.posteriors(i)
      i += 1
    }
    qLoglikelihood += x.qLoglikelihood
    this
  }

}

object StatAggregator {

/**
  * {{{StatAggregator}}} initializer
  *
  * Initializes an instance with initial statistics set as zero
  * @param k Number of components in the model
  * @param d Dimensionality of the data
 
  */
  def init(k: Int, d: Int): StatAggregator = {
    new StatAggregator(
      0.0,
      Array.fill(k)(0.0),
      Array.fill(k)(BDM.zeros[Double](d+1, d+1)))
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
      dists: Array[UpdatableGConcaveGaussian],
      optim: GMMOptimizer,
      n: Double)
      (agg: StatAggregator, y: BDV[Double]): StatAggregator = {

    val q = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.gConcavePdf(y) // <--q-logLikelihood
    }
    val qSum = q.sum
    agg.qLoglikelihood += math.log(qSum) / n
    var i = 0
    val outer = y*y.t
    while (i < agg.k) {
      q(i) /= qSum 
      agg.gradients(i) += optim.direction( dists(i), outer , q(i)) / n
      agg.posteriors(i) += q(i)
      agg.qLoglikelihood += optim.evaluateRegularizationTerm(dists(i),weights(i)) / (n*n)
      i = i + 1
    }
    agg
  }
}