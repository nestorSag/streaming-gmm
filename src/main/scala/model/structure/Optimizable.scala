package com.github.nestorsag.gradientgmm.model

import com.github.nestorsag.gradientgmm.optim.algorithms.{Optimizer}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Contains the basic functionality for an object to be modified by {{{Optimizer}}}

  */
trait Optimizable extends Serializable {

/**
  * optimizer object

  */
  protected var optimizer: Optimizer

  def setOptimizer(optim: Optimizer): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: Optimizer = optimizer

/**
  * Perform a gradient-based optimization step
  * @param data Data to fit the model
  */
  def step(data: RDD[SV]): Unit

}
