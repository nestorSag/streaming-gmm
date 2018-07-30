package com.github.gradientgmm.optim.regularization

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}

import breeze.numerics.log

import org.apache.log4j.Logger


/**
  * Conjugate prior regularization for all the mixture's parameters; this means an 
  * Inverse-Wishart prior over the covariance matrices, a Normal prior over the means
  * and a Dirchlet prior over the weights.

  * See [[https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution]]

  * See [[https://en.wikipedia.org/wiki/Dirichlet_distribution]]

  * @param dim Data dimensionality

  */
class ConjugatePrior(val dim: Int, var k: Int) extends Regularizer{

	require(dim>0,"dimensionality must be a positive integer")
	require(k>0,"number of clusters must be a positive integer")

/**
  * prior mean for components' mean vector

  */
	private var muPriorMean: BDV[Double] = BDV.zeros[Double](dim)

/**
  * prior mean for components' covariance matrix

  */
	private var sigmaPriorMean: BDM[Double] = BDM.eye[Double](dim)

/**
  * Degrees of freedom for the covariance prior

  */
	private var df: Double = dim + 1

/**
  * Concentration parameter for the weight vector prior

  */
	private var weightConcentrationPar: Double = 0.5 

/**
  * degrees of freedom for the Inverse-Wishart prior

  * @param df Degrees of freedom

  */
	def setDf(df: Double): this.type = {
		//require(df>sigmaPriorMean.cols-1,"degrees of freedom must me greater than dim(sigmaPriorMean)")
		require(df>0,"degrees of freedom must me greater than zero")
		this.df = df
		this.regularizingMatrix = buildRegMatrix(df,muPriorMean,sigmaPriorMean)
		this
	}

	def getDf = this.df

/**
  * Set mean and covariance parameters' prior means.
  * The Gaussian parameter prior means must be set at the same time to check correctness, since their dimension must match

  */
	def setGaussianPriorMeans(muPriorMean: BDV[Double], sigmaPriorMean: BDM[Double]): this.type = {
		val logger: Logger = Logger.getLogger("conjugatePrior")

		require(sigmaPriorMean.cols == sigmaPriorMean.rows, "sigma prior mean is not a square matrix")
		require(muPriorMean.length == sigmaPriorMean.cols, "parameters' dimensions does not match")
		require(sigmaPriorMean == sigmaPriorMean.t,"sigmaPriorMean must be symmetric")

		if(df <= sigmaPriorMean.cols-1){
			this.df = sigmaPriorMean.cols
			logger.info(s"Setting df to ${this.df}. It must be larger than ${muPriorMean.length}")
		}
		this.muPriorMean = muPriorMean
		this.sigmaPriorMean = sigmaPriorMean
		this.regularizingMatrix = buildRegMatrix(df,muPriorMean,sigmaPriorMean)
		this
	}

	def getMuPriorMean = muPriorMean

	def getSigmaPriorMean = sigmaPriorMean

	def setWeightConcentrationPar(alpha: Double): this.type = {
		require(alpha>0,"Dirichlet prior concentration parameter must be positive")
		this.weightConcentrationPar = alpha
		this
	}

	def getWeightConcentrationPar = weightConcentrationPar

	def setK(k: Int): this.type = {
		require(k>0,"number of clusters must be positive")
		this.k = k
		this
	}

	def getK = this.k
/**
  * Get augmented parameter matrix. See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''

  */
	var regularizingMatrix = buildRegMatrix(df,muPriorMean,sigmaPriorMean)


	def gaussianGradient(dist:UpdatableGaussianComponent): BDM[Double] = {
		val kappa = df+dim+2
		(this.regularizingMatrix - dist.paramMat * kappa)*0.5
	}

	def weightsGradient(weights: BDV[Double]): BDV[Double] = {
		(BDV.ones[Double](k) - weights*k.toDouble)*weightConcentrationPar
	}

	def evaluateDist(dist: UpdatableGaussianComponent): Double = {
		val kappa = df + dim + 2
		- 0.5*(kappa*(dist.logDetSigma + math.log(dist.getS)) + symProdTrace(regularizingMatrix,dist.invParamMat))
	}

	def evaluateWeights(weights: BDV[Double]): Double = {
		weightConcentrationPar*sum(log(weights))
	}
	
/**
  * Build augmented parameter matrix

  */
	private def buildRegMatrix(df: Double, muPriorMean: BDV[Double], sigmaPriorMean: BDM[Double]): BDM[Double] = {

		//       [sigmaPriorMean + kappa*muPriorMean*muPriorMean.t, kappa*muPriorMean
		//        kappa*muPriorMean^T                     ,         kappa]
		val kappa = df + dim + 2

		val shrinkedMu = muPriorMean*kappa
		val lastRow = new BDV[Double](shrinkedMu.toArray ++ Array(kappa))

		BDM.vertcat(BDM.horzcat(sigmaPriorMean + shrinkedMu*muPriorMean.t,shrinkedMu.toDenseMatrix.t),lastRow.asDenseMatrix)
	}

/**
  * Faster computation of Tr(X*Y) for symmetric matrices

  */
	private def symProdTrace(x: BDM[Double], y: BDM[Double]): Double = { 
		
		x.toArray.zip(y.toArray).foldLeft(0.0){case (s,(a,b)) => s + a*b}
	}
}