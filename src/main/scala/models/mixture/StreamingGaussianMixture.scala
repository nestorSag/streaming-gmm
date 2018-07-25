package com.github.gradientgmm.models

import com.github.gradientgmm.components.{UpdatableGaussianComponent, UpdatableWeights, Utils}
import com.github.gradientgmm.optim.algorithms.{Optimizable, Optimizer, GradientAscent}

import breeze.linalg.{diag, eigSym, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV, Vectors => SVS, Matrices => SMS}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.rdd.RDD


import org.apache.log4j.Logger

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
  * Creates a new StreamingGaussianMixture instance
  * @param weights Array of weights
  * @param gaussians Array of mixture components
  * @param optim Optimizer object
 
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

/**
  * Creates a new StreamingGaussianMixture instance initialized with the
  * results of a K-means model fitted with a sample of the data
  * @param data training data in the form of an RDD of Spark vectors
  * @param optim Optimizer object
  * @param k Number of components in the mixture
  * @param nSamples Number of data points to train the K-means model
  * @param nIters Number of iterations allowed for the K-means model
  * @param seed random seed
  */
  def initialize(
    data: RDD[SV],
    optim: Optimizer,
    k: Int,
    nSamples: Int,
    nIters: Int,
    seed: Long = 0): StreamingGaussianMixture = {
    
    val sc = data.sparkContext
    val d = data.take(1)(0).size
    val n = math.max(nSamples,2*k)
    var samples = sc.parallelize(data.takeSample(withReplacement = false, n, seed))

    //create kmeans model
    val kmeansModel = new KMeans()
      .setMaxIterations(nIters)
      .setK(k)
      .setSeed(seed)
      .run(samples)
    
    val means = kmeansModel.clusterCenters.map{case v => Utils.toBDV(v.toArray)}

    //add means to sample points to avoid having cluster with zero points 
    samples = samples.union(sc.parallelize(means.map{case v => SVS.dense(v.toArray)}))

    // broadcast values to compute sample covariance matrices
    val kmm = sc.broadcast(kmeansModel)
    val scMeans = sc.broadcast(means)

    // get empirical cluster proportions to initialize the mixture/s weights
    //add 1 to counts to avoid division by zero
    val proportions = samples
      .map{case s => (kmm.value.predict(s),1)}
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()
      .map{case (k,p) => p.toDouble}

    val scProportions = sc.broadcast(proportions)

    //get empirical covariance matrices
    //also add a rescaled identity matrix to avoid starting with singular matrices 
    val pseudoCov = samples
      .map{case v => {
          val prediction = kmm.value.predict(v)
          val denom = math.sqrt(scProportions.value(prediction))
          (prediction,(Utils.toBDV(v.toArray)-scMeans.value(prediction))/denom) }} // x => (x-mean)
      .map{case (k,v) => (k,v*v.t)}
      .reduceByKey(_ + _)
      .map{case (k,v) => {
        val avgVariance = math.max(1e-4,trace(v))/d
        (k,v + BDM.eye[Double](d) * avgVariance)
        }}
      .sortByKey()
      .collect()
      .map{case (k,m) => m}

    new StreamingGaussianMixture(
      new UpdatableWeights(proportions.map{case p => p/(n+k)}), 
      (0 to k-1).map{case i => UpdatableGaussianComponent(means(i),pseudoCov(i))}.toArray,
      optim)

  }

  /**
  * Fit a Gaussian Mixture Model (see [[https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model]]).
  * The model is initialized using a K-means algorithm over a small sample and then 
  * fitting the resulting parameters to the data using this {GMMOptimization} object
  * @param data Data to fit the model
  * @param optim Optimization algorithm
  * @param k Number of mixture components (clusters)
  * @param batchSize number of samples processed per iteration
  * @param maxIter maximum number of gradient ascent steps allowed
  * @param convTol log-likelihood change tolerance for stopping criteria
  * @param startingSampleSize Sample size for the K-means algorithm
  * @param kMeansIters Number of iterations allowed for the K-means algorithm
  * @param seed Random seed
  * @return Fitted model
  */
  def fit(
    data: RDD[SV], 
    optim: Optimizer = new GradientAscent(), 
    k: Int = 2,
    batchSize: Option[Int] = None,
    maxIter: Int = 100,
    convTol: Double = 1e-6, 
    startingSampleSize: Int = 50,
    kMeansIters: Int = 20, 
    seed: Int = 0): StreamingGaussianMixture = {
    
    val model = initialize(
                  data,
                  optim,
                  k,
                  startingSampleSize,
                  kMeansIters,
                  seed)
    
    if(batchSize.isDefined){
      model.setBatchSize(batchSize.get)
    }

    model
    .setMaxIter(maxIter)
    .setConvergenceTol(convTol)
    .step(data)

    model
  }
}
