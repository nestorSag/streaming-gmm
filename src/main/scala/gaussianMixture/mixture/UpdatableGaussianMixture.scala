package com.github.nestorsag.gradientgmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD


class UpdatableGaussianMixture(
  private[gradientgmm] var weights: UpdatableWeights,
  private[gradientgmm] var gaussians: Array[UpdatableGaussianMixtureComponent]) extends Serializable {


  def getWeights: Array[Double] = weights.weights
  def getGaussians: Array[UpdatableGaussianMixtureComponent] = gaussians

  def k: Int = weights.length

  private val EPS = Utils.EPS
  
  require(weights.length == gaussians.length, "Length of weight and Gaussian arrays must match")

  // Spark vector predict methods
  def predict(points: RDD[SV]): RDD[Int] = {
    val responsibilityMatrix = predictSoft(points)
    responsibilityMatrix.map(r => r.indexOf(r.max))
  }


  def predict(point: SV): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  def predict(points: JavaRDD[SV]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]


  def predictSoft(points: RDD[SV]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val bcDists = sc.broadcast(gaussians)
    val bcWeights = sc.broadcast(weights.weights)
    points.map { x =>
      computeSoftAssignments(new BDV[Double](x.toArray), bcDists.value, bcWeights.value, k)
    }
  }


  def predictSoft(point: SV): Array[Double] = {
    computeSoftAssignments(new BDV[Double](point.toArray), gaussians, weights.weights, k)
  }

  def predict(point: BDV[Double]): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  def predictSoft(point: BDV[Double]): Array[Double] = {
    computeSoftAssignments(point, gaussians, weights.weights, k)
  }

  private def computeSoftAssignments(
      pt: BDV[Double],
      dists: Array[UpdatableGaussianMixtureComponent],
      weights: Array[Double],
      k: Int): Array[Double] = {
    val p = weights.zip(dists).map {
      case (weight, dist) => EPS + weight * dist.pdf(pt) //ml eps
    }
    val pSum = p.sum
    for (i <- 0 until k) {
      p(i) /= pSum
    }
    p
  }

}
