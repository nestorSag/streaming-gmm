package com.github.gradientgmm.models

import com.github.gradientgmm.components.{UpdatableGaussianComponent, UpdatableWeights, Utils}
import com.github.gradientgmm.optim.algorithms.{Optimizable, Optimizer, GradientAscent}


import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.rdd.RDD


/**
  * Gradient-based Gaussian Mixture model for streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param w Weight vector wrapper
  * @param g Array of mixture components (distributions)
  * @param optim Optimization object
 
  */
class StreamingGaussianMixture private[models] (
  _w:  UpdatableWeights,
  _g: Array[UpdatableGaussianComponent],
  var _optim: Optimizer) extends GradientGaussianMixture(_w,_g,_optim) {

/**
  * Update model parameters using streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
	def step(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      step(rdd)
    }
	}

/**
  * Cluster membership prediction for streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
  def predict(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predict(rdd)
    }
  }

/**
  * Soft cluster membership prediction for streaming data
  * See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''
  * @param data Streaming data
 
  */
  def predictSoft(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predictSoft(rdd)
    }
	}

}



object StreamingGaussianMixture{
/**
  * Creates a new StreamingGaussianMixture instance with a GradientAscent optimizer
  * @param weights Array of weights
  * @param gaussians Array of mixture components
 
  */
  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGaussianComponent]): StreamingGaussianMixture = {

    new StreamingGaussianMixture(
      new UpdatableWeights(weights),
      gaussians,
      new GradientAscent())
  }

/**
  * Creates a new StreamingGaussianMixture instance
  * @param weights Array of weights
  * @param gaussians Array of mixture components
  * @param optim Optimizer object
 
  */
  def apply(
    weights: Array[Double],
    gaussians: Array[UpdatableGaussianComponent],
    optim: Optimizer): StreamingGaussianMixture = {

    new StreamingGaussianMixture(
      new UpdatableWeights(weights),
      gaussians,
      optim)
  }

}
