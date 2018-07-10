import org.scalatest.FlatSpec
import com.github.nestorsag.gradientgmm.{UpdatableGaussianMixtureComponent,ConjugatePrior}
import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm, det}

import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

// tests for ConjugatePrior class

class ConjugatePriorTest extends FlatSpec {
  
  val errorTol = 1e-8
  val covdim = 20

  //get random matrix

  val randA = BDM.rand(covdim,covdim)
  val cov = randA.t * randA + BDM.eye[Double](covdim) // form SPD covariance matrix

  //get random mean vector 

  val mu = BDV.rand(covdim)
  val k = 10

  def logdet(m: BDM[Double]): Double = {math.log(det(m))}

  "regularizingMatrix" should "be well-formed" in { 

    // same kind of structure as un dist.ParamMat
    
    var prior = new ConjugatePrior()
      .setDf(covdim)
      .setGaussianParsPriorMeans(mu,cov)
      .setWeightConcentrationPar(1.0/k)
      .setK(k)

    val lcv = BDV(Array.fill(covdim)(0.0) ++ Array(1.0)) // last canonical vector e_d = (0,...,0,1)

    //create reshaper matrix
    val reshaper = BDM.zeros[Double](covdim+1,covdim)
    for(i <- 0 to covdim-1){
      reshaper(i,i) = 1
    }

    val principalSubmatrix = reshaper.t * prior.regularizingMatrix * reshaper
    val matdiff = principalSubmatrix - (cov + (mu * (mu.t * covdim.toDouble)))

    assert(trace(matdiff.t * matdiff) < errorTol)

    // check that last row contains mu (concatenated with s)
    var vecdiff = (lcv.t * prior.regularizingMatrix * reshaper).t - mu * covdim.toDouble

    assert(math.pow(norm(vecdiff),2) < errorTol)

    // check that last col contains mu (concatenated with s)
    vecdiff = (reshaper.t * prior.regularizingMatrix * lcv) - mu * covdim.toDouble

    assert(math.pow(norm(vecdiff),2) < errorTol)
  }

  "evaluate()" should "return correct regularization term values" in { 


    // -df/2*log det paramMat - 0.5*trace(psi*sInv)
    var prior = new ConjugatePrior()
      .setDf(covdim)
      .setGaussianParsPriorMeans(mu,cov)
      .setWeightConcentrationPar(1.0/k)
      .setK(k)

    var testdist = UpdatableGaussianMixtureComponent(covdim,mu,cov) // when paramMat = regularizingMatrix

    //Tr(regMat*paramMat) = dim(regMat) = dim(paramMat) = covdim + 1
    //weightGrad(1.0) = 0
    var shouldBeZero = prior.evaluate(testdist,1.0) - (-0.5*prior.getDf*logdet(testdist.paramMat) - 0.5*(covdim+1))
    assert(math.pow(shouldBeZero,2) < errorTol)

    // try moving the weight 
    shouldBeZero = prior.evaluate(testdist,0.5) - (-0.5*prior.getDf*logdet(testdist.paramMat) - 0.5*(covdim+1) + prior.getWeightConcentrationPar*math.log(0.5))
    assert(math.pow(shouldBeZero,2) < errorTol)

    // when paramMat = identity, logdet should cancel out
    testdist = UpdatableGaussianMixtureComponent(BDV.zeros[Double](covdim),BDM.eye[Double](covdim))

    //logdet(paramMat) = log(1) = 0
    shouldBeZero = prior.evaluate(testdist,1.0) - (- 0.5*trace(prior.regularizingMatrix))
    assert(math.pow(shouldBeZero,2) < errorTol)


    // when paramMat = identity and regularizingMatrix = almost identity (last diagonal entry = df, which cant equal one
    // if regularization is a true conjugate prior unless the problem's dimensionality is one)
    prior = new ConjugatePrior()
      .setDf(covdim)
      .setGaussianParsPriorMeans(BDV.zeros[Double](covdim),BDM.eye[Double](covdim))
      .setWeightConcentrationPar(1.0/k)
      .setK(k)


    shouldBeZero = prior.evaluate(testdist,1.0) - (- 0.5*(covdim+testdist.getS*prior.getDf))
    assert(math.pow(shouldBeZero,2) < errorTol)
  }
  // gradient methods are simple enough

}