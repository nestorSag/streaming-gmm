package streamingGmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import breeze.numerics.sqrt

class GMMAdam(
	learningRate: Double,
	regularizer: Option[GMMRegularizer],
	var beta1: Double,
	var beta2: Double) extends GMMGradientAscent(learningRate,regularizer) {

	var t: Int = 0 //timestep

	def setBeta1(beta1: Double): Unit = { 
		require(beta1 > 0 , "beta1 must be positive")
		this.beta1 = beta1
	}

	def getBeta1: Double = { 
		this.beta1
	}

	def setBeta2(beta2: Double): Unit = { 
		require(beta2 > 0 , "beta2 must be positive")
		this.beta2 = beta2
	}

	def getBeta2: Double = { 
		this.beta2
	}


	override def direction(dist: UpdatableMultivariateGaussian, sampleInfo: BDM[Double]): BDM[Double] = {

		t += 1

		if(!dist.momentum.isDefined){
			dist.initializeMomentum
		}

		if(!dist.rsmge.isDefined){
			dist.initializeRsmge
		}

		val grad = lossGradient(dist, sampleInfo)

		dist.updateMomentum(dist.momentum.get*beta1 + grad*(1-beta1))
		
		dist.updateRsmge(dist.rsmge.get*beta2 + (grad *:* grad)*(1-beta2))

		val alpha_t = learningRate*math.sqrt(1 - math.pow(beta2,t))/(1 - math.pow(beta1,t))

		alpha_t * dist.momentum.get /:/ sqrt(dist.rsmge.get)
	}

}