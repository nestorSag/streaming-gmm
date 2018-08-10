package com.github.gradientgmm

import com.github.gradientgmm.components.{UpdatableGaussianComponent, Utils}
import com.github.gradientgmm.optim.Regularizer

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, sum}

import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

/**
  * Distributed aggregator of relevant statistics
  *
  * In each worker it computes and aggregates the current batch log-likelihood,
  * and the terms that will later be used to compte the gradient. The class structure is based on
  * Spark's [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/GaussianMixture.scala ExpectationSum]]

  * @param weightsGradient Aggregated posterior responsability for each component
  * @param posteriorsAgg Sum of posterior cluster responsibilities
  * @param outerProductsAgg Sum of weighted outer products
  * @param loss Aggregated log-likelihood
  * @param counter Batch size counter
 
  */
class MetricAggregator(
  val weightsGradient: BDV[Double],
  val posteriorsAgg: BDV[Double],
  val outerProductsAgg: Array[Array[Double]], //treat matrices as arrays for BLAS routines
  var loss: Double,
  var counter: Int) extends Serializable{

/**
  * Number of components in the model
  */
  val k = weightsGradient.length

/**
  * Adder for other MetricAggregator instances
  *
  * Used for further aggregation between each worker's object
 
  */
  val m = outerProductsAgg(0).length

  def +=(x: MetricAggregator): MetricAggregator = {

    var i = 0
    while (i < k) {

      var j = 0
      var xcurrent = x.outerProductsAgg(i)
      while(j < m){
        outerProductsAgg(i)(j) += xcurrent(j)
      }

      i += 1
    }
    posteriorsAgg += x.posteriorsAgg
    weightsGradient += x.weightsGradient
    loss += x.loss
    counter += x.counter
    this
  }

}

object MetricAggregator {

/**
  * MetricAggregator initializer
  *
  * Initializes an instance with initial statistics set as zero
  * @param k Number of components in the model
  * @param d Dimensionality of the data
 
  */
  def init(k: Int, d: Int): MetricAggregator = {
    new MetricAggregator(
      BDV.zeros[Double](k),
      BDV.zeros[Double](k),
      Array.fill(k)(Array.fill((d+1)*(d+1))(0.0)),
      0.0,
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
      dists: Array[UpdatableGaussianComponent])
      (agg: MetricAggregator, y: BDV[Double]): MetricAggregator = {

    val d = dists(0).getMu.length // data dimensionality

    agg.counter += 1

    var posteriors = getPosteriors(y,dists,weights)
    
    val vectorWeights = Utils.toBDV(weights)
    
    agg.loss += math.log(sum(posteriors))


    posteriors /= sum(posteriors)

    agg.posteriorsAgg += posteriors
    // update aggregated weight gradient

    agg.weightsGradient += (posteriors - vectorWeights) //gradient


    agg.weightsGradient(weights.length - 1) = 0.0 // last weight's auxiliar variable is fixed because of the simplex cosntraint

    // aggregate outer products
    var i = 0
    while (i < agg.k) {

      //add only upper traingle
      nativeBLAS.dsyr("U", y.length, posteriors(i), y.toArray, 1, agg.outerProductsAgg(i), d+1)

      i = i + 1
    }
    agg
  }

/**
  * compute posterior membership probabilities for a data point
  *
  * Used for reducing individual data points and aggregating their statistics
  * @param point Data point
  * @param dists Current model components
  * @param weights current model's weights
  * @return Vector of posterior membership probabilities
 
  */
  def getPosteriors(point: BDV[Double], dists: Array[UpdatableGaussianComponent], weights: Array[Double]): BDV[Double] = {
    
    val q = weights.zip(dists).map {
      case (weight, dist) =>  weight * dist.gConcavePdf(point) // <--q-logLikelihood
    }

    var p = Utils.toBDV(q)

    for(i <- 0 to p.length-1){
      p(i) = math.min(math.max(p(i),Double.MinPositiveValue),Double.MaxValue/p.length)
    }

    p

  }

  // the following code was taken from Spark's [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/linalg/BLAS.scala BLAS]]
  @transient private var _f2jBLAS: NetlibBLAS = _
  @transient private var _nativeBLAS: NetlibBLAS = _
  
  private def nativeBLAS: NetlibBLAS = {
    if (_nativeBLAS == null) {
      _nativeBLAS = NativeBLAS
    }
    _nativeBLAS
  }


}