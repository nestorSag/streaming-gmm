package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Optimizer interface that contains base hyperparameters and their getters and setters.
  * Optimization algorithms like Stochastic Gradient Ascent are implementations of this trait

  */
trait GMMOptimizer extends Serializable{

/**
  * Regularization term, if any

  */
	private[gradientgmm] val regularizer: Option[GMMRegularizer] = None

/**
  * Calculates the mapping from and to the weights' Simplex (see [[]]) and the transformation's gradient

  */
	private[gradientgmm] val weightOptimizer: GMMWeightTransformation = new SoftmaxWeightTransformation()

/**
  * Ascent procedure's learning rate

  */
	private[gradientgmm] var learningRate: Double = 0.9

/**
  * Rate at which the learning rate is decreased as the number of iterations grow.
  * After {{{t}}} iterations the learning rate will be {{{shrinkageRate^t * learningRate}}}

  */
    private[gradientgmm] var shrinkageRate: Double = 0.95

/**
  * Minimum allowed learning rate. Once this lower bound is reached the learning rate will not
  * shrink anymore

  */
	private[gradientgmm] var minLearningRate: Double = 1e-2

/**
  * Minibatch size for each iteration in the ascent procedure. If {{{None}}}, it does
  * full batch gradient ascent.

  */
	private[gradientgmm] var batchSize: Option[Int] = None

/**
  * Maximum allowed tolerance in the change in log-likelihood for the program to finish

  */
	private[gradientgmm] var convergenceTol: Double = 1e-6

/**
  * Maximum number of iterations allowed

  */
	private[gradientgmm] var maxIter: Int = 100

	def setLearningRate(learningRate: Double): this.type = { 
		require(learningRate > 0 , "learning rate must be positive")
		this.learningRate = learningRate
		this
	}

	def getLearningRate: Double = { 
		this.learningRate
	}

	def setMinLearningRate(m: Double): this.type = { 
		require(m >= 0, "minLearningRate rate must be in positive")
		minLearningRate = m
		this
	}

	def getMinLearningRate: Double = { 
		minLearningRate
	}

	def setShrinkageRate(s: Double): this.type = { 
		require(s > 0 &&  s <= 1.0, "learning rate must be in (0,1]")
		shrinkageRate = s
		this
	}

	def getBatchSize: Option[Int] = batchSize

	def setBatchSize(n: Int): this.type = {
		require(n>0,"n must be a positive integer")
		batchSize = Option(n)
		this
	}

	def getShrinkageRate: Double = { 
		shrinkageRate
	}

    def getConvergenceTol: Double = convergenceTol

	def setConvergenceTol(x: Double): this.type = {
		require(x>0,"convergenceTol must be positive")
		convergenceTol = x
		this
	}


	def setMaxIter(m: Int): this.type = {
		require(m > 0 ,s"maxIter needs to be a positive integer; got ${m}")
		maxIter = m
		this
	}

	def getMaxIter: Int = {
		maxIter
	}

/**
  * Shrink {{{learningRate}}} by {{{shrinkageRate}}}
  *
  */
	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

/**
  * Use the {{{fromSimplex}}} method from [[GMMWeightTransformation]]
  *
  * @param weights mixture weights
  */
	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.fromSimplex(weights)
	}

/**
  * Use the {{{toSimplex}}} method from [[GMMWeightTransformation]]
  *
  * @param real vector
  * @return valid mixture weight vector
  */
	def toSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.toSimplex(weights)
	}


/**
  * Computes the full loss gradient 
  */
	private[gradientgmm] def lossGradient(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double] = {

		regularizer match{
			case None => basicLossGradient(dist.paramMat,point,w) 
			case Some(_) => basicLossGradient(dist.paramMat,point,w) +
				regularizer.get.gradient(dist)
		}

	}

/**
  * Computes the loss gradient of the mixture component without regularization term 
  */
	private[gradientgmm] def basicLossGradient(paramMat: BDM[Double], point: BDM[Double], w: Double): BDM[Double] = {

		(point - paramMat) * 0.5 * w
	}

/**
  * Computes the loss gradient of the weights vector without regularization term 
  */
	private[gradientgmm] def basicSoftWeightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		weightOptimizer.gradient(posteriors,weights)
	}
/**
  * Computes the full loss gradient of the weights vector 
  */
	private[gradientgmm] def softWeightGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		var grads = regularizer match {
			case None => basicSoftWeightsGradient(posteriors,weights)
			case Some(_) => basicSoftWeightsGradient(posteriors,weights) +
		 			regularizer.get.softWeightsGradient(weights)
		}

		grads(weights.length - 1) = 0.0

		grads

	}

/**
  * Evaluate regularization term, if any, with current parameters
  *
  * @param dist Mixture component
  * @param weight Corresponding mixture weight
  * @return regularization term value
  */
	def evaluateRegularizationTerm(dist: UpdatableGConcaveGaussian,weight: Double): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluate(dist,weight)
		}

	}

/**
  * Fit a Gaussian Mixture Model (see [[https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model]]).
  * The model is initialized using a K-means algorithm over a small sample and then 
  * fitting the resulting parameters to the data using this {{{GMMOptimization}}} object

  * @param data Data to fit the model
  * @param k Number of mixture components (clusters)
  * @param startingSampleSize Sample size for the K-means algorithm
  * @param kMeansIters Number of iterations allowed for the K-means algorithm
  * @param seed Random seed
  * @return Fitted model
  */
	def fit(data: RDD[SV], k: Int = 2, startingSampleSize: Int = 50, kMeansIters: Int = 20, seed: Int = 0): GradientBasedGaussianMixture = {
		
		val model = GradientBasedGaussianMixture.initialize(
			data,
			this,
			k,
			startingSampleSize,
			kMeansIters,
			seed)
		    
		model.step(data)

		model
	}

/**
  * Compute the ascent direction 

  * @param dist Mixture component
  * @param point Data point
  * @param w {{{dist}}}'s posterior responsability for {{{point}}} (see [[StatAggregator]])
  * @return ascent direction for the component's parameters
  */
	def direction(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double]

/**
  * Compute the ascent direction for the weight vector

  * @param posteriors posterior responsability for the corresponding mixture component
  * @param weights vector fo current weights
  * @return ascent direction for weight parameters
  */
	def softWeightsDirection(posteriors: BDV[Double], weights: UpdatableWeights): BDV[Double]


}