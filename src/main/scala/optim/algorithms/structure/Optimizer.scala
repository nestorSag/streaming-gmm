package com.github.gradientgmm.optim.algorithms

import com.github.gradientgmm.optim.regularization.Regularizer
import com.github.gradientgmm.optim.weights.{WeightsTransformation,SoftmaxWeightTransformation}
import com.github.gradientgmm.components.{UpdatableGaussianComponent, AcceleratedGradientUtils}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, sum}


import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Optimizer interface that contains base hyperparameters and their getters and setters.
  * Optimization algorithms like Stochastic Gradient Ascent are implementations of this trait
  */
trait Optimizer extends Serializable{
  
/**
  * Optional regularization term
  */
	protected var regularizer: Option[Regularizer] = None

/**
  * Ascent procedure's learning rate
  */
	protected var learningRate: Double = 0.9

/**
  * Rate at which the learning rate is decreased as the number of iterations grow.
  * After t iterations the learning rate will be shrinkageRate^t * learningRate
  */
  protected var shrinkageRate: Double = 0.95

/**
  * Minimum allowed learning rate. Once this lower bound is reached the learning rate will not
  * shrink anymore
  */
	protected var minLearningRate: Double = 1e-2

/**
  * Calculates the mapping from and to the weights' Simplex (see [[https://en.wikipedia.org/wiki/Simplex]]) and the transformation's gradient
  */
  private[gradientgmm] var weightsOptimizer: WeightsTransformation = new SoftmaxWeightTransformation()

/**
  * Expected batch size. This is needed to correctly weight regularizers' contributions to
  * gradients and loss function
  */
  private var n: Double = 1.0
 /**
  * Shrink {learningRate} by {shrinkageRate}
  *
  */
	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

/**
  * Use the {fromSimplex} method from [[com.github.gradientgmm.optim.weights.WeightsTransformation]]
  *
  * @param weights mixture weights
  */
	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightsOptimizer.fromSimplex(weights)
	}

/**
  * Use the {toSimplex} method from [[com.github.gradientgmm.optim.weights.WeightsTransformation]]
  *
  * @param real vector
  * @return valid mixture weight vector
  */
	def toSimplex(weights: BDV[Double]): BDV[Double] = {
		weightsOptimizer.toSimplex(weights)
	}


/**
  * Computes the full loss gradient for Gaussian parameters
  * @param dist Mixture component
  * @param point outer product of an augmented data point x => [x 1]
  * @param w prior responsibility of {point} by {dist}
  */
	def gaussianGradient(dist: UpdatableGaussianComponent, point: BDM[Double], w: Double): BDM[Double] = {

    var grad = basicGaussianGradient(dist.paramMat,point,w)

    if(regularizer.isDefined){
      grad += regularizer.get.gaussianGradient(dist)/n
    }

    grad

	}

/**
  * Computes the loss gradient of the mixture component without regularization term 
  */
	protected def basicGaussianGradient(paramMat: BDM[Double], point: BDM[Double], w: Double): BDM[Double] = {

		(point - paramMat) * 0.5 * w
	}

/**
  * Computes the loss gradient of the weights vector without regularization term 
  */
	protected def basicWeightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		weightsOptimizer.gradient(posteriors,weights)
	}
/**
  * Computes the full loss gradient of the weights vector 
  */
	def weightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

    var grads = basicWeightsGradient(posteriors,weights)

    if(regularizer.isDefined){
      grads += regularizer.get.weightsGradient(weights)/n
    }

    grads(weights.length - 1) = 0.0 // last weight's auxiliar variable is fixed because of the simplex cosntraint

		grads

	}

/**
  * Evaluate regularization term for one of the model's components
  *
  * @param dist Mixture component
  * @return regularization term value
  */
	def evaluateRegularizationTerm(dist: UpdatableGaussianComponent): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluateDist(dist)/n
		}

	}

  /**
  * Evaluate regularization term for the model's weights vector
  *
  * @param weight weights vector
  * @return regularization term value
  */
  def evaluateRegularizationTerm(weights: BDV[Double]): Double = {

    regularizer match{
      case None => 0
      case Some(_) => regularizer.get.evaluateWeights(weights)/n
    }

  }


/**
  * Compute full updates for the model's parameters. Usually this has the form X_t + alpha * direction(X_t)
  * but it differs for some algorithms, e.g. Nesterov's gradient ascent.
  *
  * @param current Current parameter values
  * @param grad Current batch gradient
  * @param utils Wrapper for accelerated gradient ascent utilities
  * @param ops Deffinitions for algebraic operations for the apropiate data structure, e.g. vector or matrix.
  * .
  * @return updated parameter values
  */


  def getUpdate[A](current: A, grad:A, utils: AcceleratedGradientUtils[A])(implicit ops: ParameterOperations[A]): A = {
    // performing gradient ascent (not descent) despite the name of the classes
    ops.sum(current,ops.rescale(direction(grad,utils)(ops),learningRate))

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
  * Alternative method to set step size's shrinkage rate. it will be automatically calculated to shrink
  * the step size by half every n iterations.
  *
  * @param n positive intger
  * @return this
  */
  def halveStepEvery(n: Int): this.type = {
    require(n>0, "n must be a positive integer")

    shrinkageRate = math.pow(2.0,-1.0/n)
    this

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

	def getShrinkageRate: Double = { 
		shrinkageRate
	}

  def setWeightsOptimizer(t: WeightsTransformation): this.type = {
    weightsOptimizer = t
    this
  }

  def setRegularizer(r: Regularizer): this.type = {
    regularizer = Option(r)
    this
  }

  def setN(n: Double): this.type = {
    this.n = n
    this
  }

  def getN = n

  override def toString: String = {

    val reg: String = if(regularizer.isDefined){
      s"Regularization term: ${regularizer.toString}"
    }else{
      "Regularization term: None"
    }

    val output = s"Algorithm: ${this.getClass} \n ${reg} \n Weight transformation: ${weightsOptimizer.toString}"
    output
  }

}