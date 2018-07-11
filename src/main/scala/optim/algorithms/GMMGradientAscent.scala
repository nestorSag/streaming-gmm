package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

/**
  * Computes stochastic gradient ascent directions
  */
class GMMGradientAscent extends GMMOptimizer{ 

	require(learningRate>0,"learningRate must be positive")

	def weightsDirection(grad: BDV[Double], utils: AcceleratedGradientUtils[BDV[Double]]): BDV[Double] = {
		grad
	}

	def gaussianDirection(grad: BDM[Double], utils: AcceleratedGradientUtils[BDM[Double]]): BDM[Double] = {
		grad

	}

	// override def direction[T <: {def :* : Double => T; def + : T =>T}](grad: T, utils: AcceleratedGradientUtils[T]): T = {
	// 	grad
	// }

	//override  def getUpdate[T <: {def :* : Double => T; def + : T =>T}](current: T, grad: T, utils: AcceleratedGradientUtils[T]): T = {
	//	current + direction(grad,utils) * learningRate
	//}



	
}