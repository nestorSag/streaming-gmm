import streamingGmm.{GMMGradientAscent, UpdatableMultivariateGaussian}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}

// This test checks convergence in expectation on a single gaussian component 
class GradientAscentTest extends OptimTestSpec{

	var lr = 0.5
	var current = UpdatableMultivariateGaussian(BDV.rand(dim),BDM.eye[Double](dim))
	var optim = new GMMGradientAscent(lr,None)
	val paramMat0 = current.paramMat
		
	"GradientAscent w/o reg" should "make current dist converge to target dist in expectation" in {

		for(i <- 1 to niter){

			current.update(current.paramMat + optim.direction(current,targetParamMat) * optim.getLearningRate)

		}

		// for a single component, expected result after n iterations is:
		// Y + (1 - lr/2)^n * (X0 - Y)
		// where Y is the target matrix and X0 the initial guess

		val expectedMat = (targetParamMat + (paramMat0 - targetParamMat) * math.pow(1 -lr/2.0,niter))
		var diff = expectedMat - current.paramMat
		assert(trace(diff.t * diff) < errorTol)
	
	}


	it should "make current weights converge to target weights in expectation" in {

		for(i <- 1 to niter){

			targetWeightsObj.update(targetWeightsObj.soft + optim.softWeightsDirection(targetWeights,targetWeightsObj) * optim.getLearningRate)

		}

		// for a single component, expected result after n iterations is:
		// Y + (1 - lr/2)^n * (X0 - Y)
		// where Y is the target weight vector and X0 the initial guess

		val expectedWeights = (targetWeights + (initialWeights - targetWeights) * math.pow(1 -lr/2.0,niter))
		var diff = toBDV(targetWeightsObj.weights) - expectedWeights 
		assert(norm(dif) < errorTol)
	
	}



}