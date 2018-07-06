package edu.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class GMMGradientAscent(
	private[gradientgmm] var learningRate: Double,
	val regularizer: Option[GMMRegularizer]) extends GMMOptimizer{ 

	require(learningRate>0,"learningRate must be positive")

	var minLearningRate = 1e-3
	var shrinkageRate = 1.0

	def penaltyValue(dist: UpdatableMultivariateGaussian,weight: Double): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluate(dist,weight)
		}

	}

	def softWeightsDirection(posteriors: BDV[Double], weights: SGDWeights): BDV[Double] = {
		softWeightGradient(posteriors,new BDV(weights.weights))
	}

	def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		lossGradient(dist,sampleInfo)

	}

	private[gradientgmm] def basicLossGradient(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		val posteriorProb = sampleInfo(sampleInfo.rows-1,sampleInfo.cols-1)

		(sampleInfo - paramMat*posteriorProb)*0.5
	}

	private[gradientgmm] def lossGradient(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		regularizer match{
			case None => basicLossGradient(dist.paramMat,sampleInfo) 
			case Some(_) => basicLossGradient(dist.paramMat,sampleInfo) +
				regularizer.get.gradient(dist)
		}

	}


	private[gradientgmm] def basicSoftWeightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		val n = posteriors.sum

		posteriors - weights*n

		//paramMat(paramMat.rows-1,paramMat.cols-1) - n*weight
	}

	private[gradientgmm] def softWeightGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		var grads = regularizer match {
			case None => basicSoftWeightsGradient(posteriors,weights)
			case Some(_) => basicSoftWeightsGradient(posteriors,weights) +
		 			regularizer.get.softWeightsGradient(weights)
		}

		grads(weights.length - 1) = 0.0

		grads

	}
	
}