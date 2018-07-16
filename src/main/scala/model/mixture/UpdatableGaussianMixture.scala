package com.github.gradientgmm.model

import com.github.gradientgmm.components.{UpdatableGaussianComponent, UpdatableWeights, Utils}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Implementation of a Gaussian Mixture Model with updatable components. 

  * The class is strongly based on Spark's [[https://spark.apache.org/docs/2.1.1/api/scala/index.html#org.apache.spark.mllib.clustering.GaussianMixture GaussianMixture]], except it allows mutable components
  * 
  * @param w Weight vector wrapper
  * @param g Array of mixture components (distributions)
  * @param optimizer Optimization object
 
  */
class UpdatableGaussianMixture(
  private[gradientgmm] var weights: UpdatableWeights,
  private[gradientgmm] var gaussians: Array[UpdatableGaussianComponent]) extends Serializable {


  def getWeights: Array[Double] = weights.weights
  def getGaussians: Array[UpdatableGaussianComponent] = gaussians

/**
  * number of componenrs
 
  */
  def k: Int = weights.length

  private val EPS = Utils.EPS
  
  require(weights.length == gaussians.length, "Length of weight and Gaussian arrays must match")

  /**
  * Cluster membership prediction for an RDD o Spark vectors

  * @return RDD with the points' labels
 
  */
  def predict(points: RDD[SV]): RDD[Int] = {
    val responsibilityMatrix = predictSoft(points)
    responsibilityMatrix.map(r => r.indexOf(r.max))
  }

  /**
  * Cluster membership prediction for a single Spark vector

  * @return vector membership label
 
  */
  def predict(point: SV): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  /**
  * Cluster membership prediction for a JavaRDD o Spark vectors

  * @return RDD with the points' labels
 
  */
  def predict(points: JavaRDD[SV]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
  * Soft cluster membership prediction for a RDD of Spark vectors

  * @return RDD with arrays giving the membership probabilities for each cluster
 
  */
  def predictSoft(points: RDD[SV]): RDD[Array[Double]] = {
    val sc = points.sparkContext
    val bcDists = sc.broadcast(gaussians)
    val bcWeights = sc.broadcast(weights.weights)
    points.map { x =>
      computeSoftAssignments(new BDV[Double](x.toArray), bcDists.value, bcWeights.value, k)
    }
  }

  /**
  * Soft cluster membership prediction for a single Spark vector

  * @return RDD with arrays giving the membership probabilities for each cluster
 
  */
  def predictSoft(point: SV): Array[Double] = {
    computeSoftAssignments(new BDV[Double](point.toArray), gaussians, weights.weights, k)
  }

  /**
  * Soft cluster membership prediction for a single Breeze vector

  * @return vector membership label
 
  */
  def predict(point: BDV[Double]): Int = {
    val r = predictSoft(point)
    r.indexOf(r.max)
  }

  /**
  * Soft cluster membership prediction for a single Breeze vector

  * @return Array giving the membership probabilities for each cluster
 
  */
  def predictSoft(point: BDV[Double]): Array[Double] = {
    computeSoftAssignments(point, gaussians, weights.weights, k)
  }

  /**
  * process individual points to compute soft cluster assignments
 
  */
  private def computeSoftAssignments(
      pt: BDV[Double],
      dists: Array[UpdatableGaussianComponent],
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
