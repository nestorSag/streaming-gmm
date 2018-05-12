package org.uoe.sgdgmm

class InverseWishartNormalDirichletPrior private (val psi: Matrix, val rho: Double, val weigthPriorConcentration: Double) extends Prior{

	val dirichletPrior = DirichletPrior(weigthPriorConcentration)


	def evaluate(gaussians: List[GConcaveGaussian], weights: List[Double]): Double = {
		
		val gaussianTerm = gaussians.map{g => -rho/2*g.logdetS() - beta/2*accumulate(psi :*: g.invS())}.sum

		val weightTerm = dirichletPrior.evaluate(weights)

		gaussianTerm + weightTerm
	}

	def getGradient(gaussians: List[GConcaveGaussian], weights: List[Double]): List((Matrix,Double)) = {
		
		val k = weights.length
		
		val grad_s = gaussians.map{g => -rho/2*g.gConcaveSigma() + beta/2*psi.get}

		val grad_w = dirichletPrior.getGradient(weights)

		grad_s.zip(grad_w)

	}

	object InverseWishartNormalDirichletPrior {

		def apply(varPriorDf: Double, 
			      varPriorScale: Matrix, 
			      meanPriorMean: Vector, 
			      meanPriorVar: Matrix, 
			      meanPriorVarShrinkage: Double,
			      weigthPriorConcentration: Double,
			      beta: Double) = Try[InverseWishartNormalDirichletPrior]{

			val varPriorRows: Int  = varPriorScale.rows
			val varPriorCols: Int = varPriorScale.rows

			val meanLength: Int = meanPriorMean.length

			val meanVarRows: Int = meanPriorVar.rows
			val meanVarCols: Int = meanPriorVar.cols

			if(varPriorRows != varPriorCols){
				Failure("varPriorScale matrix is not square")
			}

			if(varPriorDf <= varPriorRows){
				Failure("varPriorDf need to be greather than %d".format(varPriorRows))
			}

			if(meanLength != varPriorRows){
				Failure("dimensions differ in varPriorScale and meanPriorMean")
			}

			if(meanVarRows != meanVarCols){
				Failure("meanPriorVar matrix is not square")
			}

			if(varPriorRows != meanVarRows){
				Failure("varPriorScale and meanPriorVar dimensions differ")
			}

			if(meanPriorVarShrinkage <= 0){
				Failure("meanPriorVarShrinkage must be positive")
			}

			if(weigthPriorConcentration <= 0){
				Failure("weigthPriorConcentration must be positive")
			}

			if(beta <= 0){
				Failure("beta must be positive")
			}

			// if the problem has a regularization term, the block matrix psi
			// that wraps the priors has to be built and passed to GConcaveLoss

			val k = meanPriorShrinkage.get

			val psi: Matrix = {
				val d = varPriorScale.get.numRows
				val v = varPriorDf.get
				val r = (k-1)/(d+v+1)

				val block_psi = r*varPriorScale.get.asBreeze + k*(meanPriorMean.get.asBreeze)*(meanPriorMean.get.asBreeze).t
				block_psi = DenseMatrix.vertcat(block_psi,k*(meanPriorMean.get.asBreeze).t)
				block_psi = DenseMatrix.horzcat(block_psi,k*(meanPriorMean.get ++ 1).asBreeze)
			}

			val rho = k*beta

			Success[InverseWishartNormalDirichletPrior](new InverseWishartNormalDirichletPrior(psi,rho,weigthPriorConcentration))
		}
	}
}
