package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

class GMMGradientAscent extends GMMOptimizer{ 

	require(learningRate>0,"learningRate must be positive")

	def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double] = {
		softWeightGradient(posteriors,new BDV(weights.weights))
	}

	def direction(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double] = {

		lossGradient(dist,point,w)

	}
	
}