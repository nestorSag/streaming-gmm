package net.github.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class StatAggregator(
  var qLoglikelihood: Double,
  val posteriors: Array[Double],
  val gradients: Array[BDM[Double]]) extends Serializable{

  val k = gradients.length

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

  def init(k: Int, d: Int): StatAggregator = {
    new StatAggregator(
      0.0,
      Array.fill(k)(0.0),
      Array.fill(k)(BDM.zeros[Double](d+1, d+1)))
  }

  //compute cluster contributions for each input point
  // (U, T) => U for aggregation
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