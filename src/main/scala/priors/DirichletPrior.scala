package org.uoe.sgdgmm

class DirichletPrior private (weigthPriorConcentration: Double) extends Prior{

	def evaluate(weights: List[Double]): Double = {
		
		weigthPriorConcentration*weights.map{w => math.log(w)}.sum

	}

	def getGradient(weights: List[Double]): List(Double) = {
		
		val k = weights.length
		
		val grad_w = weigths.map{w => weigthPriorConcentration*(1-k*w)}

		grad_w

	}

	object DirichletPrior {

		def apply(weigthPriorConcentration: Double) = Try[DirichletPrior]{

			if(weigthPriorConcentration <= 0){
				Failure("weigthPriorConcentration must be positive")
			}

			Success[DirichletPrior](new DirichletPrior(weigthPriorConcentration))
		}
	}
}