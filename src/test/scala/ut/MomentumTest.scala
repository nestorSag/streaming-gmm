import com.github.gradientgmm.components.UpdatableGaussianMixtureComponent
import com.github.gradientgmm.optim.algorithms.MomentumGradientAscent

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}



/**
  * Check correct trajectories for gradient ascent in expectation
  * this means testing the procedure with a single-valued sample that would represent the mean of 
  * the actual samples
  */
class MomentumTest extends OptimTestSpec{

	var lr = 0.5
	var beta = 0.9
	var current = UpdatableGaussianMixtureComponent(BDV.rand(dim),BDM.eye[Double](dim))
	var optim = new MomentumGradientAscent().setLearningRate(lr).setBeta(beta)

	"MomentumGradientAscent w/o reg" should "follow the right path in expectation to target Gaussian parameters" in {
		
		var x0 = current.paramMat.copy

		var m = BDM.zeros[Double](x0.rows,x0.cols) //momentum
		
		//calculate gradient descent with momentum in expectation
		// this will be checked against the program's results below
		for(i <- 1 to niter){
			var g = (targetParamMat - x0) * 0.5 //gradient
			m *= beta
			m += g
			x0 += m*lr
		}

		for(i <- 1 to niter){

			//current.update(current.paramMat + optim.direction(current,targetParamMat, 1.0) * optim.getLearningRate)
			current.update(
				optim.getGaussianUpdate(
					current.paramMat,
					optim.gaussianGradient(current,targetParamMat,1.0),
					current.optimUtils))
		}

		// result should be 
		// S_0 + alpha*sum((1-beta^(niters+1-i)/(1-beta))*grad(S_i))

		var diff =  x0 - current.paramMat
		assert(trace(diff.t * diff) < errorTol)
	
	}

	it should "follow the right path in expectation to target weights" in {

		// deterministic formula for Momentum descent in expectation
		var x0 = toBDV(initialWeights.toArray)
		var m = BDV.zeros[Double](x0.length) //momentum

		var softx0 = toBDV(x0.toArray.map{case w => math.log(w/x0(k-1))})

		//calculate gradient descent with momentum in expectation
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