import org.scalatest.{FunSuite}


import com.github.gradientgmm.components.UpdatableGaussianComponent
import com.github.gradientgmm.GradientGaussianMixture
import com.github.gradientgmm.optim.GradientAscent

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.linalg.{Matrices => SMS, Matrix => SM, DenseMatrix => SDM, Vector => SV, Vectors => SVS, DenseVector => SDV}

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, norm, trace, det}

// tests for UpdatableGaussianMixture methods
// needs to be a running spark local cluster
trait SparkTester extends FunSuite{
    var sc : SparkContext = _

    val conf = new SparkConf().setAppName("GradientGaussianMixture-test").setMaster("local")

    val errorTol = 1e-8
    val k = 3

    sc = new SparkContext(conf)
    val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
    val parsedData = data.map(s => SVS.dense(s.trim.split(' ').map(_.toDouble))).cache()

    val mygmm = GradientGaussianMixture.init(parsedData,k)

    val weights = mygmm.getWeights
    val gaussians = mygmm.getGaussians.map{
    case g => new MultivariateGaussian(
    SVS.dense(g.getMu.toArray),
    SMS.dense(g.getSigma.rows,g.getSigma.cols,g.getSigma.toArray))}

    val sparkgmm = new GaussianMixtureModel(weights,gaussians)

    val x = parsedData.take(1)(0)

}

class GaussianMixtureTests extends SparkTester{

  try{
	  test("predict() should give same result as spark GMM model for single vector") {

	    assert(sparkgmm.predict(x) - mygmm.predict(x) == 0)
	  }

	  test("predict() should give same result as spark GMM model for RDDs"){

	    val res = sparkgmm.predict(parsedData).zip(mygmm.predict(parsedData)).map{case (x,y) => (x-y)*(x-y)}.sum

	    assert(res == 0)

	  }

	  test("predictSoft() should give same result as spark GMM model for single vector"){
	    var v1 = new BDV(sparkgmm.predictSoft(x))
	    var v2 = new BDV(mygmm.predictSoft(x))

	    assert(norm(v1-v2)*norm(v1-v2) < errorTol)
	  }

	  test("predictSoft() should give same result as spark GMM model for RDDs") {
	    
	    val res = sparkgmm.predictSoft(parsedData).zip(mygmm.predictSoft(parsedData)).map{case (a,b) => {
	     val x = new BDV(a)
	     val y = new BDV(b)
	     norm(x-y)*norm(x-y)
	     }}.sum

	    assert(res < errorTol)
	    sc.stop()
	  }
	} catch{
		case _: Throwable => println("SOmething went wrong")
	}
	// finally{
	// 	sc.stop() //doesn't work. fixing this is in todo list
	// }

  // the step() method will be tested in the integration testing stage
}
