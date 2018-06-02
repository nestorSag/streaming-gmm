package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GMMRegularizer extends Serializable {

	def gradient(paramMat: BDM[Double]): BDM[Double]

	def updateRegularizer(paramMat: BDM[Double]): Unit

	def weightGradient(weight: Double): Double

}

class ConjugatePrior(
	val df: Double, 
	val priorMu: BDV[Double], 
	val priorSigma: BDM[Double], 
	val weightPrior: Double, 
	val numClusters: Int) extends GMMRegularizer {

	require(df>priorSigma.cols-1,"degrees of freedom must me greater than dim(priorSigma)")
	require(weightPrior>0,"Dirichlet prior concentration must be positive")

	val regularizingMatrix: BDM[Double] = {

		//       [priorSigma, df*priorMu
		//        df*priorMu^T,       1]
		
		val shrinkedMu = priorMu*df
		val lastRow = new BDM[Double](1,shrinkedMu.length+1,shrinkedMu.toArray ++ Array(df))

		BDM.vertcat(BDM.horzcat(priorSigma,shrinkedMu.toDenseMatrix.t),lastRow)
	}

	def gradient(paramMat: BDM[Double]): BDM[Double] = {
		this.regularizingMatrix
	}

	def updateRegularizer(paramMat: BDM[Double]): Unit = {
		// priors are fixed
	}

	def weightGradient(weight: Double): Double = {
		weightPrior - numClusters*weightPrior*weight
	}
}