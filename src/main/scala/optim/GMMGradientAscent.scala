package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class GMMGradientAscent(
	var learningRate: Double,
	val regularizer: Option[GMMRegularizer]) extends Serializable{ 

	var weightLearningRate = 0.8

	def setWeightLearningRate(x: Double): Unit = {
		require(x>0,"x should be positive")
		weightLearningRate = x
	}

	def setLearningRate(learningRate: Double): Unit = { 
		require(learningRate > 0 , "learning rate must be positive")
		this.learningRate = learningRate
	}

	def getLearningRate: Double = { 
		this.learningRate
	}

	def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		lossGradient(dist,sampleInfo)

	}

	def penaltyValue(dist: UpdatableMultivariateGaussian,weight: Double): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluate(dist,weight)
		}

	}

	private def basicLossGradient(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		val posteriorProb = sampleInfo(sampleInfo.rows-1,sampleInfo.cols-1)

		(sampleInfo - paramMat*posteriorProb)*0.5
	}

	private def basicWeightGradient(paramMat: BDM[Double], weight: Double, n: Double): Double = {

		paramMat(paramMat.rows-1,paramMat.cols-1) - n*weight
	}

	def weightGradient(sampleInfo: BDM[Double], weight: Double, n: Double, isLastCluster: Boolean): Double = {

		if(isLastCluster){
			0
		}else{
			regularizer match{
				case None => basicWeightGradient(sampleInfo,weight,n)
				case Some(_) => basicWeightGradient(sampleInfo,weight,n) +
					regularizer.get.weightGradient(weight)
			}
		}

	}

	def lossGradient(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		regularizer match{
			case None => basicLossGradient(dist.paramMat,sampleInfo) 
			case Some(_) => basicLossGradient(dist.paramMat,sampleInfo) +
				regularizer.get.gradient(dist)
		}

	}
	
}