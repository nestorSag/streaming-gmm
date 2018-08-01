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
	private var normalMean: BDV[Double] = BDV.zeros[Double](dim)

/**
  * prior mean for components' covariance matrix

  */
	private var iwMean: BDM[Double] = BDM.eye[Double](dim)

/**
  * Degrees of freedom for the covariance prior

  */
	private var df: Double = dim

/**
  * Concentration parameter for the weight vector prior

  */
	private var dirichletParam: Double = 1.0/k

/**
  * degrees of freedom for the Inverse-Wishart prior

  * @param df Degrees of freedom

  */
	def setDf(df: Double): this.type = {
		//require(df>iwMean.cols-1,"degrees of freedom must me greater than dim(iwMean)")
		require(df>dim-1,"degrees of freedom must me greater than dim-1")
		this.df = df
		this.regularizingMatrix = buildRegMatrix(df,normalMean,iwMean)
		this
	}

	def getDf = this.df

/**
  * Set mean and covariance parameters' prior means. 
  * The Gaussian parameter prior means must be set at the same time to check correctness, since their dimension must match
  
  * @param normalMean Expected value vector for the prior Normal distribution
  * @param iwMean Expected value matrix for the prior Inverse-Wishart distribution
  */
	def setMeanAndCovExpVals(normalMean: BDV[Double], iwMean: BDM[Double]): this.type = {
		val logger: Logger = Logger.getLogger("conjugatePrior")

		require(iwMean.cols == iwMean.rows, "sigma prior mean is not a square matrix")
		require(normalMean.length == iwMean.cols, "parameters' dimensions does not match")
		require(iwMean == iwMean.t,"iwMean must be symmetric")

		if(df <= iwMean.cols-1){
			this.df = iwMean.cols
			logger.info(s"Setting df to ${this.df}. It must be larger than ${normalMean.length}")
		}
		this.normalMean = normalMean
		this.iwMean = iwMean
		this.regularizingMatrix = buildRegMatrix(df,normalMean,iwMean)
		this
	}

	def getNormalMean = normalMean

	def getIwMean = iwMean

	def setDirichletParam(alpha: Double): this.type = {
		require(alpha>0,"Dirichlet prior concentration parameter must be positive")
		this.dirichletParam = alpha
		this
	}

	def getDirichletParam = dirichletParam

	def setK(k: Int): this.type = {
		require(k>0,"number of clusters must be positive")
		this.k = k
		this
	}

	def getK = this.k
/**
  * Get augmented parameter matrix. See ''Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization''

  */
	var regularizingMatrix = buildRegMatrix(df,normalMean,iwMean)


	def gaussianGradient(dist:UpdatableGaussianComponent): BDM[Double] = {
		val kappa = df+dim+2
		(this.regularizingMatrix - dist.paramMat * kappa)*0.5
	}

	def weightsGradient(weights: BDV[Double]): BDV[Double] = {
		(BDV.ones[Double](k) - weights*k.toDouble)*dirichletParam
	}

	def evaluateDist(dist: UpdatableGaussianComponent): Double = {
		val kappa = df + dim + 2
		- 0.5*(kappa*(dist.logDetSigma + math.log(dist.getS)) + symProdTrace(regularizingMatrix,dist.invParamMat))
	}

	def evaluateWeights(weights: BDV[Double]): Double = {
		dirichletParam*sum(log(weights))
	}
	
/**
  * Build augmented parameter matrix

  */
	private def buildRegMatrix(df: Double, normalMean: BDV[Double], iwMean: BDM[Double]): BDM[Double] = {

		//       [iwMean + kappa*normalMean*normalMean.t, kappa*normalMean
		//        kappa*normalMean^T                     ,         kappa]
		val kappa = df + dim + 2

		val shrinkedMu = normalMean*kappa
		val lastRow = new BDV[Double](shrinkedMu.toArray ++ Array(kappa))

		BDM.vertcat(BDM.horzcat(iwMean + shrinkedMu*normalMean.t,shrinkedMu.toDenseMatrix.t),lastRow.asDenseMatrix)
	}

/**
  * Faster computation of Tr(X*Y) for symmetric matrices

  */
	private def symProdTrace(x: BDM[Double], y: BDM[Double]): Double = { 
		
		x.toArray.zip(y.toArray).foldLeft(0.0){case (s,(a,b)) => s + a*b}
	}
}