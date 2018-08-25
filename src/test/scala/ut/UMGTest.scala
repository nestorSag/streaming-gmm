import org.scalatest.FlatSpec

import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

// tests for UpdatableGaussianComponent class

class UMGTest extends FlatSpec {
  
  val errorTol = 1e-8
  val covdim = 20

  //get random matrix

  val randA = BDM.rand(covdim,covdim)
  val cov = randA.t * randA + BDM.eye[Double](covdim) // form SPD covariance matrix

  //get random mean vector 

  val mu = BDV.rand(covdim)

  var umgdist = UpdatableGaussianComponent(mu,cov)
  val mldist = new MultivariateGaussian(SVS.dense(mu.toArray),SMS.dense(covdim,covdim,cov.toArray))

  val btp = BDV.rand(covdim) //breeze test point
  val stp = SVS.dense(btp.toArray) //spark test point

  "for a breeze vector x, pdf(x)" should "give the same result thant MLlib's MultivariateGaussian (up to some rounding error)" in {
  	val dif = math.abs(umgdist.pdf(btp) - mldist.pdf(stp))
  	assert(math.pow(dif,2) < errorTol)
  }

  "for a Spark vector x, pdf(x)" should "give the same result thant MLlib's MultivariateGaussian (up to some rounding error)" in {
  	val dif = math.abs(umgdist.pdf(stp) - mldist.pdf(stp))
  	assert(math.pow(dif,2) < errorTol)
  }

  "for a breeze vector x, logpdf(x)" should "give the same result thant MLlib's MultivariateGaussian (up to some rounding error)" in {
  	val dif = math.abs(umgdist.logpdf(btp) - mldist.logpdf(stp))
  	assert(math.pow(dif,2) < errorTol)
  }

  "for a Spark vector x, logpdf(x)" should "give the same result thant MLlib's MultivariateGaussian (up to some rounding error)" in {
  	val dif = math.abs(umgdist.logpdf(stp) - mldist.logpdf(stp))
  	assert(math.pow(dif,2) < errorTol)
  }

  "for y = [x 1], gConcavePdf(y)" should "give the same result thant MLlib's MultivariateGaussian pdf(x) (up to some rounding error)" in {
    val dif = math.abs(umgdist.gConcavePdf(BDV(stp.toArray ++ Array(1.0)) - mldist.pdf(stp)))
    assert(math.pow(dif,2) < errorTol)
  }

  "logDetSigma" should "give the same result than Breeze's det function" in { 
    val dif = math.abs(umgdist.logDetSigma - math.log(det(umgdist.getSigma)))
    assert(math.pow(dif,2) < errorTol)

  }

  "detSigma" should "give the same result than Breeze's det function" in { 
    val dif = math.abs(umgdist.detSigma - det(umgdist.getSigma))
    assert(math.pow(dif,2) < errorTol)

  }


  "the parameter matrix" should "be well-formed" in {
  	val paramBlockMatrix = umgdist.paramBlockMatrix
  	
  	val lcv = BDV(Array.fill(covdim)(0.0) ++ Array(1.0)) // last canonical vector e_d = (0,...,0,1)

  	// check that last diagonal element equals s

  	assert(math.pow(lcv.t * paramBlockMatrix * lcv - umgdist.getS,2) < errorTol)

  	//check that the principal submatrix is well-formed

  	//create reshaper matrix
  	val reshaper = BDM.zeros[Double](covdim+1,covdim)
  	for(i <- 0 to covdim-1){
  		reshaper(i,i) = 1
  	}

  	val principalSubmatrix = reshaper.t * umgdist.paramBlockMatrix * reshaper
  	val matdiff = principalSubmatrix - (umgdist.getSigma + umgdist.getMu * umgdist.getMu.t * umgdist.getS)

  	assert(trace(matdiff.t * matdiff) < errorTol)

  	// check that last row contains mu (concatenated with s)
  	var vecdiff = (lcv.t * paramBlockMatrix * reshaper).t - umgdist.getMu

  	assert(math.pow(norm(vecdiff),2) < errorTol)

  	// check that last col contains mu (concatenated with s)
    vecdiff = (reshaper.t * paramBlockMatrix * lcv) - umgdist.getMu

  	assert(math.pow(norm(vecdiff),2) < errorTol)
  }

  "invparamBlockMatrix" should "give paramBlockMatrix inverse" in {
    
    var shouldBeZeroMat = umgdist.paramBlockMatrix*umgdist.invParamBlockMatrix - BDM.eye[Double](covdim+1)

    assert(trace(shouldBeZeroMat.t*shouldBeZeroMat) < errorTol)
    
  }

  "update()" should "correctly map updated matrix to new params" in {
    
    val paramBlockMatrix = umgdist.paramBlockMatrix.copy
    // var newdist = umgdist.update(paramBlockMatrix)

    // val shouldBeZeroMat = paramBlockMatrix - newdist.paramBlockMatrix
    umgdist.update(paramBlockMatrix)
    val dif = paramBlockMatrix - umgdist.paramBlockMatrix
    //assert(true)
    assert(trace(dif.t*dif) < errorTol)
    
  }

  // the step() method will be tested in the integration testing stage

}