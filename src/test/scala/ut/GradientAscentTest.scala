import com.github.nestorsag.gradientgmm.{GMMGradientAscent, UpdatableGaussianMixtureComponent}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}

// This test checks convergence in expectation on a single gaussian component 
class GradientAscentTest extends OptimTestSpec{

	var lr = 0.5
	var current = UpdatableGaussianMixtureComponent(BDV.rand(dim),BDM.eye[Double](dim))
	var optim = new GMMGradientAscent().setLearningRate(lr)
	val paramMat0 = current.paramMat
		
	"GradientAscent w/o reg" should "make current dist converge to target dist in expectation" in {

		//println(current.paramMat)
		for(i <- 1 to niter){
			//println(trace((current.paramMat-targetParamMat)*(current.paramMat-targetParamMat)))
			current.update(current.paramMat + optim.direction(current,targetParamMat, 1.0) * optim.getLearningRate)

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
		var beta = 0.0
		var x0 = toBDV(initialWeights.toArray)
		var m = BDV.zeros[Double](x0.length) //momentum

		var softx0 = toBDV(x0.toArray.map{case w => math.log(w/x0(k-1))})

		//calculate gradient descent in expectation
		// this will be checked against the program's results below
		for(i <- 1 to niter){
			var g = (targetWeights - x0) //gradient
			g(k-1) = 0.0
			m *= beta
			m += g
			softx0 += m*lr

			var expsoftx0 = softx0.toArray.map{case w => math.exp(w)}
			x0 = toBDV(expsoftx0.map{case w => w/expsoftx0.sum})
		}

		// get results from program
		for(i <- 1 to niter){

			var currentWeights = optim.fromSimplex(new BDV(weightObj.weights))
			var delta = optim.weightsDirection(targetWeights,weightObj) * optim.getLearningRate
			weightObj.update(optim.toSimplex(currentWeights + delta))

		}

		var vecdiff =  x0 - toBDV(weightObj.weights)
		assert(norm(vecdiff) < errorTol)
	
	}



}