import streamingGmm.{GMMAdam, UpdatableMultivariateGaussian}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}

// This test checks convergence of the Adam optimizer for a single gaussian component in expectation
class AdamTest extends OptimTestSpec{

	// since there is no simple formula to calculate the expected result externally
	// the test just makes sure that the optimizer progress steadily toward the solution
	// i.e. it doesn't diverge or oscilate wildly
	"Adam w/o reg" should "make steady progress toward target gaussian parameters" in {

		var lr = 1
		var beta1 = 0.9
		var beta2 = 0.999

		var current = UpdatableMultivariateGaussian(BDV.rand(dim),BDM.eye[Double](dim))
		var optim = new GMMAdam(lr,None,beta1,beta2)
		val paramMat0 = current.paramMat

		var expectedRes = current.paramMat.copy

		var niter = 5


		for(j <- 1 to 5){

			var diff =  targetParamMat - current.paramMat
			var previousError = trace(diff.t * diff)
			//println(s"previous error: ${previousError}")
			for(i <- 1 to niter){

				current.update(current.paramMat + optim.direction(current,targetParamMat) * optim.getLearningRate)

			}

			var newdiff = targetParamMat - current.paramMat
			assert(trace(newdiff.t * newdiff) < previousError)

		}

	
	}

	it should "make steady progress toward target weights" in {

		for(j <- 1 to 5){

			var previousError = norm(targetWeights - toBDV(targetWeightsObj.weights))
			//println(s"previous error: ${previousError}")
			
			for(i <- 1 to niter){

				targetWeightsObj.update(targetWeightsObj.soft + optim.softWeightsDirection(targetWeights,targetWeightsObj) * optim.getLearningRate)

			}

			assert(norm(targetWeights - toBDV(targetWeightsObj.weights)) < previousError)

		}

	
	}



}