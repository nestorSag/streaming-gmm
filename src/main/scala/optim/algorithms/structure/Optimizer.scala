package com.github.gradientgmm.optim

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
  * Step size
  */
	protected var learningRate: Double = 0.9

/**
  * Rate at which the learning rate is decreased as the number of iterations grow.
  * After t iterations the learning rate will be shrinkageRate^t * learningRate
  */
  protected var shrinkageRate: Double = math.pow(2.0,-1.0/20)

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
  * Shrink learningRate by a factor of shrinkageRate
  *
  */
	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

/**
  * Use fromSimplex method from [[com.github.gradientgmm.optim.WeightsTransformation WeightsTransformation]]
  *
  * @param weights mixture weights
  */
	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightsOptimizer.fromSimplex(weights)
	}

/**
  * Use toSimplex method from [[com.github.gradientgmm.optim.WeightsTransformation WeightsTransformation]]
  *
  * @param real vector
  * @return valid mixture weight vector
  */
	def toSimplex(weights: BDV[Double]): BDV[Double] = {
		weightsOptimizer.toSimplex(weights)
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
  def halveStepSizeEvery(m: Int): this.type = {
    require(m>0, "m must be a positive integer")

    shrinkageRate = math.pow(2.0,-1.0/m)
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

  def setWeightsOptimizer(wo: WeightsTransformation): this.type = {
    weightsOptimizer = wo
    this
  }

}