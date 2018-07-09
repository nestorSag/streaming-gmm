package com.github.nestorsag.gradientgmm

import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD


trait Optimizable extends Serializable {

  private[gradientgmm] var optimizer: GMMOptimizer

  def setOptimizer(optim: GMMGradientAscent): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: GMMOptimizer = optimizer

  def step(data: RDD[SV]): Unit

}
