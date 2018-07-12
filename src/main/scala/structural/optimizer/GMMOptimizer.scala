package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, sum}
import breeze.numerics.sqrt


import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Optimizer interface that contains base hyperparameters and their getters and setters.
  * Optimization algorithms like Stochastic Gradient Ascent are implementations of this trait
  */
trait GMMOptimizer extends Serializable{

/**
  * Optional regularization term
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
  * Minibatch size for each iteration in the ascent procedure. If {{{None}}}, it performs
  * full-batch optimization
  */
	private[gradientgmm] var batchSize: Option[Int] = None

/**
  * Error tolerance in log-likelihood for the stopping criteria
  */
	private[gradientgmm] var convergenceTol: Double = 1e-6

/**
  * Maximum number of iterations allowed
  */
	private[gradientgmm] var maxIter: Int = 100

/**
  * Linear Algebra operations necessary for computing updates for the parameters
    
  * This is to avoid duplicating code for Gaussian and Weights updates in the optimization
  * algorithms' classes
 
  */
  val vectorOps = new ParameterOperations[BDV[Double]] {
    def sum(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x + y}
    def sumScalar(x: BDV[Double], z: Double): BDV[Double] = {x + z}
    def rescale(x: BDV[Double], z: Double): BDV[Double] = {x*z}
    def sub(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x - y}

    def ewProd(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x *:* y}
    def ewDiv(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x /:/ y}
    def ewSqrt(x:BDV[Double]): BDV[Double] = {sqrt(x)}
  }

  val matrixOps = new ParameterOperations[BDM[Double]] {
    def sum(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x + y}
    def sumScalar(x: BDM[Double], z: Double): BDM[Double] = {x + z}
    def rescale(x: BDM[Double], z: Double): BDM[Double] = {x*z}
    def sub(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x - y}

    def ewProd(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x *:* y}
    def ewDiv(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x /:/ y}
    def ewSqrt(x:BDM[Double]): BDM[Double] = {sqrt(x)}
  }

 /**
  * Shrink {learningRate} by {shrinkageRate}
  *
  */
	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

/**
  * Use the {fromSimplex} method from [[GMMWeightTransformation]]
  *
  * @param weights mixture weights
  */
	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.fromSimplex(weights)
	}

/**
  * Use the {toSimplex} method from [[GMMWeightTransformation]]
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
	def gaussianGradient(dist: UpdatableGaussianMixtureComponent, point: BDM[Double], w: Double): BDM[Double] = {

		regularizer match{
			case None => basicGaussianGradient(dist.paramMat,point,w) 
			case Some(_) => basicGaussianGradient(dist.paramMat,point,w) +
				regularizer.get.gradient(dist)
		}

	}

/**
  * Computes the loss gradient of the mixture component without regularization term 
  */
	private[gradientgmm] def basicGaussianGradient(paramMat: BDM[Double], point: BDM[Double], w: Double): BDM[Double] = {

		(point - paramMat) * 0.5 * w
	}

/**
  * Computes the loss gradient of the weights vector without regularization term 
  */
	private[gradientgmm] def basicWeightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		weightOptimizer.gradient(posteriors,weights)
	}
/**
  * Computes the full loss gradient of the weights vector 
  */
	def weightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		var grads = regularizer match {
			case None => basicWeightsGradient(posteriors,weights)
			case Some(_) => basicWeightsGradient(posteriors,weights) +
		 			regularizer.get.weightsGradient(weights)
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
	def evaluateRegularizationTerm(dist: UpdatableGaussianMixtureComponent,weight: Double): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluate(dist,weight)
		}

	}

/**
  * Compute full updates for the weights. Usually this has the form X_t + alpha * direction(X_t)
  * but it differs for some algorithms, e.g. Nesterov's gradient ascent.
  *
  * @param current Current parameter values
  * @param grad Current batch gradient
  * @param utils Wrapper for accelerated gradient ascent utilities
  * @param ops Deffinitions for algebraic operations for the apropiate data structure, e.g. vector or matrix.
  * .
  * @return updated parameter values
  */


	def getWeightsUpdate(current: BDV[Double], grad:BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {
		
		toSimplex(fromSimplex(current) + direction(grad,utils)(vectorOps))

	}


/**
  * Compute full updates for the Gaussian parameters. Usually this has the form X_t + alpha * direction(X_t)
  * but it differs for some algorithms, e.g. Nesterov's gradient ascent.
  *
  * @param current Current parameter values
  * @param grad Current batch gradient
  * @param utils Wrapper for accelerated gradient ascent utilities
  * @param ops Deffinitions for algebraic operations for the apropiate data structure, e.g. vector or matrix.
  * .
  * @return updated parameter values
  */


	def getGaussianUpdate(current: BDM[Double], grad:BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {
		
		current + direction(grad,utils)(matrixOps)

	}

/**
  * compute the ascent direction.
  *
  * @param grad Current batch gradient
  * @param utils Wrapper for accelerated gradient ascent utilities
  * @param ops Deffinitions for algebraic operations for the apropiate data structure, e.g. vector or matrix.
  * .
  * @return updated parameter values
  */

	def direction[A](grad:A, utils: AcceleratedGradientUtils[A])(ops: ParameterOperations[A]): A


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



}


/**
  * Contains common mathematical operations that can be performed in both matrices and vectors.
  * Its purpose is avoid duplicating code in the optimization algorithms' classes
  */
trait ParameterOperations[A] extends Serializable{

  def sum(x: A, y: A): A
  def sumScalar(x:A,z:Double): A
  def rescale(x: A, d: Double): A
  def sub(x:A, y:A): A

  def ewProd(x:A,y:A): A
  def ewDiv(x:A,y:A): A
  def ewSqrt(x:A): A
}