// can't figure out how to test this properly

// import net.github.gradientgmm.{GMMAdam, UpdatableMultivariateGaussian}

// import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}

// // This test checks convergence of the Adam optimizer for a single gaussian component in expectation
// class AdamTest extends OptimTestSpec{

// 	// since there is no simple formula to calculate the expected result externally
// 	// the test just makes sure that the optimizer progress steadily toward the solution
// 	// i.e. it doesn't diverge or oscilate wildly

// 	var lr = 0.1
// 	var beta1 = 0.9
// 	var beta2 = 0.999

// 	var current = UpdatableMultivariateGaussian(BDV.rand(dim),BDM.eye[Double](dim))
// 	var optim = new GMMAdam(lr,None,beta1,beta2)

// 	"Adam w/o reg" should "make steady progress toward target gaussian parameters" in {

// 		val paramMat0 = current.paramMat

// 		var expectedRes = current.paramMat.copy

// 		for(j <- 1 to 5){

// 			var diff =  targetParamMat - current.paramMat
// 			var previousError = trace(diff.t * diff)
// 			println(s"previous error: ${previousError}")
// 			for(i <- 1 to niter){

// 				current.update(current.paramMat + optim.direction(current,targetParamMat) * optim.getLearningRate)

// 			}

// 			var newdiff = targetParamMat - current.paramMat
// 			assert(trace(newdiff.t * newdiff) < previousError)

// 		}

	
// 	}

// 	it should "make steady progress toward target weights" in {

// 		for(j <- 1 to 5){

// 			var previousError = norm(targetWeights - toBDV(weightObj.weights))
// 			//println(s"previous error: ${previousError}")
			
// 			for(i <- 1 to niter){

// 				weightObj.update(weightObj.soft + optim.softWeightsDirection(targetWeights,weightObj) * optim.getLearningRate)

// 			}

// 			assert(norm(targetWeights - toBDV(weightObj.weights)) < previousError)

// 		}

	
// 	}



// }