package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

private[streamingGmm] class SampleAggregator(
  var qLoglikelihood: Double,
  val gConcaveCovariance: Array[BDM[Double]]) extends Serializable{

  val k = gConcaveCovariance.length

  def +=(x: SampleAggregator): SampleAggregator = {
    var i = 0
    while (i < k) {
      gConcaveCovariance(i) += x.gConcaveCovariance(i)
      i += 1
    }
    qLoglikelihood += x.qLoglikelihood
    this
  }

}

private[streamingGmm] object SampleAggregator {

  def zero(k: Int, d: Int): SampleAggregator = {
    new SampleAggregator(0.0,Array.fill(k)(BDM.zeros[Double](d+1, d+1)))
  }

  //compute cluster contributions for each input point
  // (U, T) => U for aggregation
  def add(
      weights: Array[Double],
      dists: Array[UpdatableMultivariateGaussian])
      (agg: SampleAggregator, y: BDV[Double]): SampleAggregator = {

    val q = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.gConcavePdf(y) // <--q-logLikelihood
    }
    val qSum = q.sum
    agg.qLoglikelihood += math.log(qSum)
    var i = 0
    while (i < agg.k) {
      q(i) /= qSum 
      agg.gConcaveCovariance(i) += y*y.t*q(i)
      i = i + 1
    }
    agg
  }
}