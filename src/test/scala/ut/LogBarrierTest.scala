import org.scalatest.FlatSpec
import streamingGmm.{UpdatableMultivariateGaussian,LogBarrier}
import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, det, trace}


// tests for LogBarrier class
// only testing for shift = 0

class LogBarrierTest extends FlatSpec {
  
  val errorTol = 1e-8
  val covdim = 20

  var r = scala.util.Random

  var scale_ = r.nextFloat.toDouble

  var logbarrier = new LogBarrier(scale=scale_)

  val randA = BDM.rand(covdim,covdim)
  val cov = randA.t * randA + BDM.eye[Double](covdim) // form SPD covariance matrix

  //get random mean vector 

  val mu = BDV.rand(covdim)

  val umgdist = UpdatableMultivariateGaussian(mu,cov)

  val unitdist = UpdatableMultivariateGaussian(BDV.zeros[Double](covdim),BDM.eye[Double](covdim))// zero mean-unit variance dist
  
  "gradient()" should "give correct gradient" in { 

    var e = BDV(Array.fill(covdim)(0.0) ++ Array(1.0))
    var shouldBeZero = logbarrier.gradient(unitdist) - (BDM.eye[Double](covdim+1) - e * e.t) * scale_

    assert(trace(shouldBeZero.t*shouldBeZero) < errorTol)

    e = umgdist.paramMat(::,umgdist.paramMat.cols - 1)
    shouldBeZero = logbarrier.gradient(umgdist) - (umgdist.paramMat - e * e.t) * scale_

    assert(trace(shouldBeZero.t*shouldBeZero) < errorTol)

  }

  "evaluate()" should "return correct regularization term values" in { 

    var reg = logbarrier.evaluate(umgdist,1.0)

    var diff = reg - scale_ * (math.log(det(umgdist.paramMat)) - umgdist.getS)

    assert(math.pow(diff,2) < errorTol)
    
  }
  // gradient methods are simple enough

}