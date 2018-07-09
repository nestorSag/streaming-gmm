package com.github.nestorsag.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}

/**
  * Conjugate prior regularization for all the mixture's parameters. 
  * See [[https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution]]
  * See [[https://en.wikipedia.org/wiki/Dirichlet_distribution]]

  * @param df Degrees of freedom for the Inverse-Wishart prior over the covariance matrices
  * @param priorMu mean for the prior normal distribution over the means
  * @param priorSigma expected covariance matrix for the Inverse-Wishart prior over the covariance matrices
  * @param weightConcentration COncentration parameter for the Dirichlet prior over the weight vector
  * @param numClusters number of mixture components

  */
class ConjugatePrior(
	val df: Double, 
	priorMu: BDV[Double], 
	priorSigma: BDM[Double], 
	val weightConcentration: Double, 
	val numClusters: Int) extends GMMRegularizer {
	
/**
  * Get augmented parameter matrix. See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''

  */
	val regularizingMatrix = buildRegMatrix(df,priorMu,priorSigma)

	require(df>priorSigma.cols-1,"degrees of freedom must me greater than dim(priorSigma)")
	require(weightConcentration>0,"Dirichlet prior concentration parameter must be positive")

	require(priorSigma == priorSigma.t,"priorSigma must be symmetric")


	def gradient(dist:UpdatableGConcaveGaussian): BDM[Double] = {
		(this.regularizingMatrix - df*dist.paramMat)*0.5
		//updateRegularizer(paramMat)
	}

	def softWeightsGradient(weights: BDV[Double]): BDV[Double] = {
		(BDV.ones[Double](numClusters) - weights*numClusters.toDouble)*weightConcentration
	}

	def evaluate(dist: UpdatableGConcaveGaussian, weight: Double): Double = {
		evaluateGaussian(dist) + evaluateWeight(weight)
	}

/**
  * Evaluate regularization term of current component parameters

  */
	private def evaluateGaussian(dist:UpdatableGConcaveGaussian): Double = {
		- 0.5*(df*(dist.logDetSigma + math.log(dist.getS)) + symProdTrace(regularizingMatrix,dist.invParamMat))
	}

/**
  * Evaluate regularization term of current component's corresponding weight parameter

  */
	private def evaluateWeight(weight: Double): Double = {
		weightConcentration*math.log(weight)
	}
	
/**
  * Build augmented parameter matrix

  */
	private def buildRegMatrix(df: Double, priorMu: BDV[Double], priorSigma: BDM[Double]): BDM[Double] = {

		//       [priorSigma + df*priorMu*priorMu.t, df*priorMu
		//        df*priorMu^T                     ,         df]
		
		val shrinkedMu = priorMu*df
		val lastRow = new BDV[Double](shrinkedMu.toArray ++ Array(df))

		BDM.vertcat(BDM.horzcat(priorSigma + shrinkedMu*priorMu.t,shrinkedMu.toDenseMatrix.t),lastRow.asDenseMatrix)
	}

/**
  * Faster computation of Tr(X*Y) for symmetric matrices

  */
	private def symProdTrace(x: BDM[Double], y: BDM[Double]): Double = { 
		
		x.toArray.zip(y.toArray).foldLeft(0.0){case (s,(a,b)) => s + a*b}
	}
}