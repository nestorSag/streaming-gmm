import org.scalatest.FlatSpec

import com.github.gradientgmm.components.UpdatableGaussianComponent
import com.github.gradientgmm.optim.LogBarrier

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, det, trace}


// tests for LogBarrier class

class LogBarrierTest extends FlatSpec {
  
  val errorTol = 1e-8
  val covdim = 20

  var r = scala.util.Random

  var scale = r.nextFloat.toDouble + 1.0

  var logbarrier = new LogBarrier().setScale(scale)

  val randA = BDM.rand(covdim,covdim)
  val cov = randA.t * randA + BDM.eye[Double](covdim) // form SPD covariance matrix

  //get random mean vector 

  val mu = BDV.rand(covdim)

  val umgdist = UpdatableGaussianComponent(mu,cov)

  val unitdist = UpdatableGaussianComponent(BDV.zeros[Double](covdim),BDM.eye[Double](covdim))// zero mean-unit variance dist
  
  "gradient()" should "give correct gradient" in { 

    var e = BDV(Array.fill(covdim)(0.0) ++ Array(1.0))
    var shouldBeZero = logbarrier.gaussianGradient(unitdist) - (BDM.eye[Double](covdim+1) - e * e.t) * scale

    assert(trace(shouldBeZero.t*shouldBeZero) < errorTol)

    e = umgdist.paramBlockMatrix(::,umgdist.paramBlockMatrix.cols - 1)
    shouldBeZero = logbarrier.gaussianGradient(umgdist) - (umgdist.paramBlockMatrix - e * e.t) * scale

    assert(trace(shouldBeZero.t*shouldBeZero) < errorTol)

  }

  "evaluate()" should "return correct regularization term values" in { 

    var reg = logbarrier.evaluateDist(umgdist) + logbarrier.evaluateWeights(new BDV(Array(1.0))) // evaluate dist and weight separately

    var diff = reg - scale * (math.log(det(umgdist.paramBlockMatrix)) - umgdist.getS)

    assert(math.pow(diff,2) < errorTol)
    
  }
  // gradient methods are simple enough

}