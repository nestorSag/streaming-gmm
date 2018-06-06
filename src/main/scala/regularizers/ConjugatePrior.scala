package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}

class ConjugatePrior(
	val df: Double, 
	val priorMu: BDV[Double], 
	val priorSigma: BDM[Double], 
	val weightPrior: Double, 
	val numClusters: Int) extends GMMRegularizer {

	require(df>priorSigma.cols-1,"degrees of freedom must me greater than dim(priorSigma)")
	require(weightPrior>0,"Dirichlet prior concentration parameter must be positive")

	def gradient(dist:UpdatableMultivariateGaussian): BDM[Double] = {
		this.regularizingMatrix*0.5
		//updateRegularizer(paramMat)
	}

	def weightGradient(weight: Double): Double = {
		weightPrior - numClusters*weightPrior*weight
	}

	def evaluate(dist: UpdatableMultivariateGaussian, weight: Double): Double = {
		- 0.5*(dist.detSigma + dist.getS + trace(regularizingMatrix*dist.paramMat)) + weightPrior*math.log(weight)
	}

	private val regularizingMatrix: BDM[Double] = {

		//       [priorSigma, df*priorMu
		//        df*priorMu^T,       1]
		
		val shrinkedMu = priorMu*df
		val lastRow = new BDM[Double](1,shrinkedMu.length+1,shrinkedMu.toArray ++ Array(df))

		BDM.vertcat(BDM.horzcat(priorSigma,shrinkedMu.toDenseMatrix.t),lastRow)
	}

	private def updateRegularizer(paramMat: BDM[Double]): Unit = {
		// priors are fixed
	}
}