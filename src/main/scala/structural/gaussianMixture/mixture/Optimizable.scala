package com.github.nestorsag.gradientgmm

import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Contains the basic functionality for an object to be modified by {{{Optimizer}}}

  */
trait Optimizable extends Serializable {

/**
  * optimizer object

  */
  private[gradientgmm] var optimizer: Optimizer

  def setOptimizer(optim: GradientAscent): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: Optimizer = optimizer

/**
  * Perform a gradient-based optimization procedure
  * @param data Data to fit the model
  */
  def step(data: RDD[SV]): Unit

}
