package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

class GMMMomentumGradientAscent(
	learningRate: Double,
	regularizer: Option[GMMRegularizer],
	var decayRate: Double) extends GMMGradientAscent(learningRate,regularizer) {

	def setDecayRate(decayRate: Double): Unit = { 
		require(decayRate > 0 , "learning rate must be positive")
		this.decayRate = decayRate
	}

	def getDecayRate(learningRate: Double): Double = { 
		this.decayRate
	}

	override def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		if(!dist.momentum.isDefined){
			dist.momentum = Option(BDM.zeros[Double](dist.getMu.length+1,dist.getMu.length+1))
		}

		val grad = lossGradient(dist, sampleInfo)

		dist.momentum.map{ case m => m*decayRate + grad}
		
		dist.momentum.get
	}

}