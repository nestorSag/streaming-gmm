package edu.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}

class ConjugatePrior(
	val df: Double, 
	priorMu: BDV[Double], 
	priorSigma: BDM[Double], 
	val weightConcentration: Double, 
	val numClusters: Int) extends GMMRegularizer {

	// val df = degFreedom
	// val weightConcentration = dirichletParam
	// val numClusters = nClust
	val regularizingMatrix = buildRegMatrix(df,priorMu,priorSigma)

	require(df>priorSigma.cols-1,"degrees of freedom must me greater than dim(priorSigma)")
	require(weightConcentration>0,"Dirichlet prior concentration parameter must be positive")

	require(priorSigma == priorSigma.t,"priorSigma must be symmetric")


	def gradient(dist:UpdatableMultivariateGaussian): BDM[Double] = {
		(this.regularizingMatrix - df*dist.paramMat)*0.5
		//updateRegularizer(paramMat)
	}

	def softWeightsGradient(weights: BDV[Double]): BDV[Double] = {
		(BDV.ones[Double](numClusters) - weights*numClusters.toDouble)*weightConcentration
	}

	def evaluate(dist: UpdatableMultivariateGaussian, weight: Double): Double = {
		evaluateGaussian(dist) + evaluateWeight(weight)
	}

	private def evaluateGaussian(dist:UpdatableMultivariateGaussian): Double = {
		- 0.5*(df*(dist.logDetSigma + math.log(dist.getS)) + symProdTrace(regularizingMatrix,dist.invParamMat))
	}

	private def evaluateWeight(weight: Double): Double = {
		weightConcentration*math.log(weight)
	}
	
	private def buildRegMatrix(df: Double, priorMu: BDV[Double], priorSigma: BDM[Double]): BDM[Double] = {

		//       [priorSigma + df*priorMu*priorMu.t, df*priorMu
		//        df*priorMu^T                     ,         df]
		
		val shrinkedMu = priorMu*df
		val lastRow = new BDV[Double](shrinkedMu.toArray ++ Array(df))

		BDM.vertcat(BDM.horzcat(priorSigma + shrinkedMu*priorMu.t,shrinkedMu.toDenseMatrix.t),lastRow.asDenseMatrix)
	}

	private def symProdTrace(x: BDM[Double], y: BDM[Double]): Double = { // faster computation of Tr(A*B) for symmetric matrices
		
		x.toArray.zip(y.toArray).foldLeft(0.0){case (s,(a,b)) => s + a*b}
	}
}