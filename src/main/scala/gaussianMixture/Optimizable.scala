package streamingGmm

import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD


trait Optimizable extends Serializable {

  private[streamingGmm] var optimizer: GMMOptimizer

  var maxGradientIters = 100

  var convergenceTol = 1e-6


  def getConvergenceTol: Double = convergenceTol

  def setConvergenceTol(x: Double): this.type = {
    require(x>0,"convergenceTol must be positive")
    convergenceTol = x
    this
  }


  def setmaxGradientIters(maxGradientIters: Int): this.type = {
    require(maxGradientIters > 0 ,s"maxGradientIters needs to be a positive integer; got ${maxGradientIters}")
    this.maxGradientIters = maxGradientIters
    this
  }

  def getmaxGradientIters: Int = {
    maxGradientIters
  }


  def setOptimizer(optim: GMMGradientAscent): this.type = {
    optimizer = optim
    this
  }

  def getOpimizer: GMMOptimizer = optimizer


  def setLearningRate(alpha: Double): this.type = {
    require(alpha > 0,
      s"learning rate must be positive; got ${alpha}")
    optimizer.setLearningRate(alpha)
    this
  }

  def getLearningRate: Double = optimizer.learningRate


  def step(data: RDD[SV]): Unit

}
