import org.scalatest.{FlatSpec, BeforeAndAfter}

import streamingGmm.{UpdatableMultivariateGaussian, SGDGMM, GMMGradientAscent}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.{SparkConf, SparkContext}

class SGDGMMTest extends FlatSpec {

  val errorTol = 1e-8
  val k = 10
  val dim  = 10
  var nvectors = 20

  val master = "local[4]"
  val appName = "SGDGMM-test"

  var sc: SparkContext = _

  val conf = new SparkConf().setMaster(master).setAppName(appName)

  sc = new SparkContext(conf)


  // initialize weight vector
  var weights = BDV.rand(k).toArray
  weights = weights.map{x => x/weights.sum}

  // cov matrix creator
  def covGenerator: BDM[Double] = {
    val randA = BDM.rand(dim,dim)
    randA.t * randA + BDM.eye[Double](dim) // form SPD covariance matrix
  }

  // initialize array of dists
  val dists = (1 to k).map{ x => UpdatableMultivariateGaussian(BDV.rand(dim),covGenerator)}.toArray

  // instantiate the two versions of gmm

  var myGmm = SGDGMM(weights,dists,new GMMGradientAscent(0.5,None))

  var sparkGmm = new GaussianMixtureModel(
    weights,
    dists.map{
      d => new MultivariateGaussian(SVS.dense(d.getMu.toArray),SMS.dense(d.getSigma.rows,d.getSigma.cols,d.getSigma.toArray))})
  
  // create random vector for evaluation
  var x = BDV.rand(dim)

  var sparkx = SVS.dense(x.toArray)

  var bdvectors = (1 to nvectors).map{case x => BDV.rand(dim)} 

  var bdvrdd = sc.parallelize(bdvectors) // Breeze dense vectors

  var svrdd = sc.parallelize(bdvectors.map{case x => SVS.dense(x.toArray)}) // spark vectors

  "predict(x)" should "give the same result than Spark's GaussianMixtureModel" in {

    var diff = myGmm.predict(sparkx) - sparkGmm.predict(sparkx)

    assert(diff == 0)
  }

  "predict(x)" should "give the same result than Spark's GaussianMixtureModel for RDDs" in {

    var diff = myGmm.predict(svrdd).zip(sparkGmm.predict(svrdd)).map{case (x,y) => (x-y)*(x-y)}.reduce(_ + _)

    assert(diff == 0)
  }
2
  "predictSoft(x)" should "give the same result than Spark's GaussianMixtureModel" in {

    var diff = myGmm.predictSoft(sparkx).zip(sparkGmm.predictSoft(sparkx)).map{case (x,y) => (x-y)*(x-y)}.reduce(_ + _)

    assert(diff < errorTol)
  }

  "predictSoft(x) (x as Spark vector)" should "give the same result than Spark's GaussianMixtureModel for RDDs" in {

    var diff = myGmm.predictSoft(svrdd).zip(sparkGmm.predictSoft(svrdd)).map{case (x,y) => {
      var bx = new BDV(x)
      var by = new BDV(y)
      norm(bx-by)}
      }.reduce(_ + _)

    assert(diff < errorTol)
  }


  if (sc != null) {
      sc.stop()
  }
  
}