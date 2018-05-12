package org.uoe.sgdgmm

import breeze.linalg.{DenseVector => BreezeVector, DenseMatrix}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.util.{Loader, MLUtils, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

//import math.log
class SGDGaussianMixtureModel(weights: Array[Double], gaussians: Array[MultivariateGaussian]) extends GaussianMixtureModel(weights, gaussians){

	def update(data: RDD[Vector]) = {
		
		// get transformed weights
		val last_weight: Double = weights.last
		val w = weights.map(alpha => log(alpha/last_weight)) // eq. 3.7

		// transform to vector y = [x 1]
		val y = data.map(x => x :+ 1)

		// get matrix S
		val s: Array[DenseMatrix] = gaussians.map(g => {
			val s = g.sigma + g.mu * g.mu.t
			s = DenseMatrix.vertcat(s,g.mu.t)
			s = DenseMatrix.horzcat(s,mu :+ 1)
		}) // eq.3.4 with s = s* = 1


		// compute gradient

		// call stepsize = line-search(sigma,grad)

		// sigma += stepsize*gradient


	}

	def loss(y: RDD[Vector],w: Array[Double], q: Array[MultivariateGaussian]): Double = {

	}
}
