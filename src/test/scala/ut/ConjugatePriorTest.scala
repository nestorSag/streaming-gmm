import org.scalatest.FlatSpec

import com.github.gradientgmm.components.UpdatableGaussianComponent
import com.github.gradientgmm.optim.regularization.ConjugatePrior

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm, det}

import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

// tests for ConjugatePrior class

// "easy" values were selected as the prior parameters and the correct values
// were calculated by hand
class ConjugatePriorTest extends FlatSpec {
  
  val errorTol = 1e-8
  val dim = 20

  //get random matrix

  val randA = BDM.rand(dim,dim)
  val cov = randA.t * randA + BDM.eye[Double](dim) // form SPD covariance matrix

  //get random mean vector 

  val mu = BDV.rand(dim)
  val k = 10

  def logdet(m: BDM[Double]): Double = {math.log(det(m))}

  val df = dim

  val kappa = (df + dim + 2).toDouble

  "regularizingMatrix" should "be well-formed" in { 

    // same kind of structure as dist.ParamMat
    
    var prior = new ConjugatePrior(dim,k)
      .setDf(df)
      .setGaussianPriorMeans(mu,cov)
      .setWeightConcentrationPar(1.0/k)

    val lcv = BDV(Array.fill(dim)(0.0) ++ Array(1.0)) // last canonical vector e_d = (0,...,0,1)

    //create reshaper matrix
    val reshaper = BDM.zeros[Double](dim+1,dim)
    for(i <- 0 to dim-1){
      reshaper(i,i) = 1
    }

    val principalSubmatrix = reshaper.t * prior.regularizingMatrix * reshaper
    val matdiff = principalSubmatrix - (cov + (mu * (mu.t * kappa)))

    assert(trace(matdiff.t * matdiff) < errorTol)

    // check that last row contains mu*kappa
    var vecdiff = (lcv.t * prior.regularizingMatrix * reshaper).t - mu * kappa

    assert(math.pow(norm(vecdiff),2) < errorTol)

    // check that last col contains mu*kappa
    vecdiff = (reshaper.t * prior.regularizingMatrix * lcv) - mu * kappa

    assert(math.pow(norm(vecdiff),2) < errorTol)
  }

  "evaluate()" should "return correct regularization term values" in { 

    // kappa = df + dim + 2
    //regularization term: 
    // r = kappa *logdet(S) - tr(Psi * S^{-1})

    // -df/2*log det paramMat - 0.5*trace(psi*sInv)
    var prior = new ConjugatePrior(dim,k)
      .setDf(df)
      .setGaussianPriorMeans(mu,cov)
      .setWeightConcentrationPar(1.0/k)

    var testdist = UpdatableGaussianComponent(kappa,mu,cov) // when paramMat = regularizingMatrix

    val vectorWeight = new BDV(Array(1.0))

    var shouldBe = (-0.5*(prior.getDf+dim+2)*logdet(testdist.paramMat) - 0.5*(dim+1))

    var shouldBeZero = prior.evaluateDist(testdist) + prior.evaluateWeights(vectorWeight) - shouldBe
    
    assert(math.pow(shouldBeZero,2) < errorTol)

    // try moving the weight 
    shouldBe = (-0.5*(prior.getDf+dim+2)*logdet(testdist.paramMat) - 0.5*(dim+1) + prior.getWeightConcentrationPar*math.log(0.5))

    shouldBeZero = prior.evaluateDist(testdist) + prior.evaluateWeights(vectorWeight*0.5) - shouldBe
    
    assert(math.pow(shouldBeZero,2) < errorTol)

    // when paramMat = identity, logdet should cancel out
    testdist = UpdatableGaussianComponent(BDV.zeros[Double](dim),BDM.eye[Double](dim))

    //logdet(paramMat) = log(1) = 0
    shouldBe = (- 0.5*trace(prior.regularizingMatrix))

    shouldBeZero = prior.evaluateDist(testdist) + prior.evaluateWeights(vectorWeight) - shouldBe
    
    assert(math.pow(shouldBeZero,2) < errorTol)


    // when paramMat = identity and regularizingMatrix = almost identity (last diagonal entry = kappa, which cant equal one
    // if regularization is a true conjugate prior unless the problem's dimensionality is one)
    prior = new ConjugatePrior(dim,k)
      .setDf(dim)
      .setGaussianPriorMeans(BDV.zeros[Double](dim),BDM.eye[Double](dim))
      .setWeightConcentrationPar(1.0/k)


    shouldBe = (- 0.5*(dim+testdist.getS*kappa))

    shouldBeZero = prior.evaluateDist(testdist) + prior.evaluateWeights(vectorWeight) - shouldBe
    
    assert(math.pow(shouldBeZero,2) < errorTol)
  }
  // gradient methods are simple enough

}