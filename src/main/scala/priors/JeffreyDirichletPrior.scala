package org.uoe.sgdgmm

class JeffreyDirichletPrior private (val weigthPriorConcentration: Double) extends Prior{

	val dirichletPrior = DirichletPrior(weigthPriorConcentration)

	def evaluate(gaussians: List[GConcaveGaussian], weights: List[Double]): Double = {
		
		val gaussianTerm = gaussians.map{g => {g.logdetS() - beta/2*accumulate(psi :*: g.invS())

			val d = g.invS().nCols
			g.logdetS() - g.invS()(d-1,d-1)

		}}.sum

		val weightTerm = dirichletPrior.evaluate(weights)

		gaussianTerm + weightTerm
	}

	def getGradient(gaussians: List[GConcaveGaussian], weights: List[Double]): List((Matrix,Double)) = {
		
		val k = weights.length
		
		val grad_s = gaussians.map{g => {

			val s = g.gConcaveSigma()
			val d = s.nCols
			s(d-1,d-1) += 1
			s

		}}

		val grad_w = dirichletPrior.getGradient(weights)

		grad_s.zip(grad_w)

	}

	object JeffreyDirichletPrior {

		def apply(weigthPriorConcentration: Double) = Try[JeffreyDirichletPrior]{

			if(weigthPriorConcentration <= 0){
				Failure("weigthPriorConcentration must be positive")
			}

			// if the problem has a regularization term, the block matrix psi
			// that wraps the priors has to be built and passed to GConcaveLoss

			Success[JeffreyDirichletPrior](new JeffreyDirichletPrior(psi,rho,weigthPriorConcentration))
		}
	}
}
