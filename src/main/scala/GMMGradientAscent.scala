package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class GMMGradientAscent(
	var learningRate: Double, 
	val regularizer: Option[GMMRegularizer]) extends Serializable{ 

	def setLearningRate(learningRate: Double): Unit = { 
		require(learningRate > 0 , "learning rate must be positive")
		this.learningRate = learningRate
	}

	def getLearningRate(learningRate: Double): Double = { 
		this.learningRate
	}

	def basicLossGradient(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		val posteriorProb = paramMat(paramMat.rows,paramMat.cols)

		(sampleInfo - paramMat*posteriorProb)*0.5
	}

	def lossGradient(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		regularizer match{
			case None => basicLossGradient(paramMat,sampleInfo) 
			case Some(_) => basicLossGradient(paramMat,sampleInfo) +
				regularizer.get.gradient(paramMat)
		} 
	}

	def basicWeightGradient(paramMat: BDM[Double], weight: Double, n: Double): Double = {

		paramMat(paramMat.rows-1,paramMat.cols-1) - n*weight
	}

	def weightGradient(paramMat: BDM[Double], weight: Double, n: Double): Double = {

		regularizer match{
			case None => basicWeightGradient(paramMat,weight,n)
			case Some(_) => basicWeightGradient(paramMat,weight,n) +
				regularizer.get.weightGradient(weight)
		}

	}

	def direction(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		lossGradient(paramMat, sampleInfo)

	}

	
}



class GMMMomentumGradientAscent(
	learningRate: Double,
	regularizer: Option[GMMRegularizer] ,
	var decayRate: Double,
	var correction: BDM[Double]) extends GMMGradientAscent(learningRate,regularizer) {


	def setInitialCorrection(x: BDM[Double]): Unit = { 
		this.correction = x
	}

	def setDecayRate(decayRate: Double): Unit = { 
		require(decayRate > 0 , "learning rate must be positive")
		this.decayRate = decayRate
	}

	def getDecayRate(learningRate: Double): Double = { 
		this.decayRate
	}

	override def direction(paramMat: BDM[Double], sampleInfo: BDM[Double]): BDM[Double] = {

		val grad = lossGradient(paramMat, sampleInfo)

		this.correction *= decayRate
		this.correction += grad*(1-decayRate)
		this.correction
	}

}
