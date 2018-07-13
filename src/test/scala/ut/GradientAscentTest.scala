import com.github.nestorsag.gradientgmm.optim.algorithms.GradientAscent
import com.github.nestorsag.gradientgmm.components.UpdatableGaussianMixtureComponent

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}


/**
  * Check correct trajectories for gradient ascent in expectation
  * this means testing the procedure with a single-valued sample that would represent the mean of 
  * the actual samples
  */

class GradientAscentTest extends OptimTestSpec{

	var lr = 0.5
	var current = UpdatableGaussianMixtureComponent(BDV.rand(dim),BDM.eye[Double](dim))
	var optim = new GradientAscent().setLearningRate(lr)
	val paramMat0 = current.paramMat
		
	"GradientAscent" should "make current dist converge to target dist in expectation" in {

		//println(current.paramMat)
		for(i <- 1 to niter){
			//println(trace((current.paramMat-targetParamMat)*(current.paramMat-targetParamMat)))
			current.update(
				optim.getGaussianUpdate(
					current.paramMat,
					optim.gaussianGradient(current,targetParamMat,1.0),//(targetParamMat - current.paramMat) * 0.5,
					current.optimUtils))
			//current.update(current.paramMat + optim.direction((targetParamMat-current)*0.5,) * optim.getLearningRate)

		}

		// for a single component, expected result after n iterations is:
		// Y + (1 - lr/2)^n * (X0 - Y)
		// where Y is the target matrix and X0 the initial guess

		val expectedMat = (targetParamMat + (paramMat0 - targetParamMat) * math.pow(1 -lr/2.0,niter))
		var diff = expectedMat - current.paramMat
		assert(trace(diff.t * diff) < errorTol)
	
	}


	it should "make current weights converge to target weights in expectation" in {

		// the formula from above cannot be applied to the weights because of the 
		// nonlinearity induced by the softmax function

		// below: deterministic calculation for gradient descent in expectation
		// same as momentum gradient descent but with beta = 0
		var x0 = toBDV(initialWeights)

		var softx0 = optim.fromSimplex(x0)

		//calculate gradient descent in expectation
		// this will be checked against the program's results below
		for(i <- 1 to niter){
			var g = (targetWeights - x0) //gradient
			g(k-1) = 0.0 //k-1 free parameters due to restriction to to simplex
			softx0 += g*lr

			x0 = optim.toSimplex(softx0)
		}

		// get results from program
		for(i <- 1 to niter){

			var currentWeights = new BDV(weightObj.weights)
			//var delta = optim.direction(targetWeights,weightObj) * optim.getLearningRate
			//weightObj.update(optim.toSimplex(currentWeights + delta))
			weightObj.update(
				optim.getWeightsUpdate(
					currentWeights,
					optim.weightsGradient(targetWeights,currentWeights),
					weightObj.optimUtils))

		}

		var vecdiff =  x0 - toBDV(weightObj.weights)
		assert(norm(vecdiff) < errorTol)
	
	}



}