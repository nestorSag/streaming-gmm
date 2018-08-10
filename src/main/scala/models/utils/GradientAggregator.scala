package com.github.gradientgmm

import com.github.gradientgmm.components.{UpdatableGaussianComponent, Utils}
import com.github.gradientgmm.optim.regularizers.Regularizer

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, sum}

/**
  * Distributed aggregator of relevant statistics
  *
  * In each worker it computes and aggregates the current batch log-likelihood,
  * the regularization values for the current parameters and the 
  * gaussianGradients for each data point. The class structure is based heavily on
  * Spark's [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/GaussianMixture.scala ExpectationSum]]

  * @param loss aggregate log-likelihood
  * @param weightsGradient: aggregate posterior responsability for each component
  * @param gaussianGradients Aggregate point-wise gaussianGradients for each component
 
  */
class GradientAggregator(
  var loss: Double,
  val weightsGradient: BDV[Double],
  val gaussianGradients: Array[BDM[Double]],
  var counter: Int) extends Serializable{

/**
  * Number of components in the model
  */
  val k = gaussianGradients.length

/**
  * Adder for other GradientAggregator instances
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
    loss += x.loss
    counter += x.counter
    this
  }

}

object GradientAggregator {

/**
  * GradientAggregator initializer
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
      dists: Array[UpdatableGaussianComponent],
      reg: Option[Regularizer],
      n: Double)
      (agg: GradientAggregator, y: BDV[Double]): GradientAggregator = {

    agg.counter += 1

    var posteriors = getPosteriors(y,dists,weights)
    
    val vectorWeights = Utils.toBDV(weights)
    
    agg.loss += math.log(sum(posteriors))

    // add regularization value due to weights vector
    if(reg.isDefined){
      agg.loss += reg.get.evaluateWeights(vectorWeights)/n
    }

    posteriors /= sum(posteriors)
    // update aggregated weight gradient

    agg.weightsGradient += (posteriors - vectorWeights) //gradient

    // evaluate weight regularization gradient
    if(reg.isDefined){
      agg.weightsGradient += reg.get.weightsGradient(vectorWeights)/n
    }

    agg.weightsGradient(weights.length - 1) = 0.0 // last weight's auxiliar variable is fixed because of the simplex cosntraint

    // update gaussian parameters' gradients and log-likelihood
    var i = 0
    val outer = y*y.t
    while (i < agg.k) {

      agg.gaussianGradients(i) += (outer - dists(i).paramMat) * 0.5 * posteriors(i) //gradient

      if(reg.isDefined){
        agg.gaussianGradients(i) += reg.get.gaussianGradient(dists(i))/n
      }

      // add regularization value due to Gaussian components
      if(reg.isDefined){
        agg.loss += reg.get.evaluateDist(dists(i))/n
      }

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


}