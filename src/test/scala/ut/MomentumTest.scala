import streamingGmm.{GMMMomentumGradientAscent, UpdatableMultivariateGaussian}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}

// This test checks convergence in expectation on a single gaussian component 
class MomentumTest extends OptimTestSpec{

	"MomentumGradientAscent w/o reg" should "make current dist converge to target dist in expectation" in {
		var lr = 0.5
		var current = UpdatableMultivariateGaussian(BDV.rand(dim),BDM.eye[Double](dim))
		var optim = new GMMMomentumGradientAscent(lr,None,0.9)
		val paramMat0 = current.paramMat

		var expectedRes = current.paramMat.copy

		// deterministic formula for Momentum descent in expectation
		for(i <- 0 to (niter-1)){

			expectedRes += (targetParamMat-expectedRes) * 0.5 * (1.0-math.pow(optim.decayRate,niter-i))/(1.0-optim.decayRate)*optim.learningRate
			// (targetParamMat - expectedRes) * 0.5 = gradient
		}

		for(i <- 1 to niter){

			current.update(current.paramMat + optim.direction(current,targetParamMat) * optim.learningRate)

		}

		// result should be 
		// S_0 + alpha*sum((1-beta^(niters+1-i)/(1-beta))*grad(S_i))

		var diff =  expectedRes - current.paramMat
		assert(trace(diff.t * diff) < errorTol)
	
	}



}