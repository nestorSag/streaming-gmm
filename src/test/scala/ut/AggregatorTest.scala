import org.scalatest.FlatSpec

import com.github.gradientgmm.components.Utils
import com.github.gradientgmm.optim.GradientAscent
import com.github.gradientgmm.MetricAggregator
import com.github.gradientgmm.components.UpdatableGaussianComponent

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}

/**
  * This small test creates a Mixture model with two components 
  * whose weights are 0.5,0.5 and are centered at (1,0), (-1,0)
  * with identity covariance matrices.
  * We generate several 'sample' points equal to (0,1) and test the gradient and log-likelihood calulcations.
  * this toy case is easy to analyze and the correct results were derived with pen and paper.
  */
class AggregatorTest extends FlatSpec{
	
	
	var dim = 2
	var nPoints = 5
	var errorTol = 1e-8
	var k = 2
	val clusterMeans = Array(new BDV(Array(-1.0,0.0)), new BDV(Array(1.0,0.0)))
	val clusterWeights = Array(0.5,0.5)
	val clusterVars = Array.fill(2)(BDM.eye[Double](dim))

	//val mixture = GradientGaussianMixture(clusterWeights,clusterDists)
	val clusterDists = clusterMeans.zip(clusterVars).map{ case(m,v) => UpdatableGaussianComponent(m,v)}

	val targetPoint = new BDV(Array(0.0,1.0))
	// do y = [x 1]
	val points = Array.fill(nPoints)(targetPoint).map{case v => new BDV(v.toArray ++ Array(1.0))}

	val adder = MetricAggregator.add(clusterWeights, clusterDists)_
	
	val agg = points.foldLeft(MetricAggregator.init(2,dim)){case (agg,point) => adder(agg,point)}

	// copy of GradientGaussianMixture's completeMatrix method
	def completeMatrix(x: Array[Double]): BDM[Double] = {

	    // get size
	    val d = math.sqrt(x.length).toInt

	    //convert to matrix
	    val mat = new BDM(d,d,x)
	    //fill
	    mat += mat.t
	    //adjust diagonal elements
	    var i = 0
	    while(i < d){
	      mat(i,i) /= 2
	      i+=1
	    }

	    mat

  	}

  	val outerProductsMats = agg.outerProductsAgg.map(completeMatrix(_))
  	
	"the counter" should "be equal to nPoints" in {
		//weightsGradient should be zero
		assert(agg.counter == nPoints)
	}

	"the log-likelihood" should "be correclty calculated" in {
		val correctValue = (-1.0 -math.log(2*math.Pi))
		val avgLoss = agg.loss/nPoints

		var error = math.pow(avgLoss - correctValue,2)
		assert(error < errorTol)
	}

	"the posterior membership probabilities" should "be correclty calculated" in {
		//we use weightsGradient to test this
		//weightsGradient should be zero
		val avgWeightGrad = agg.weightsGradient/nPoints.toDouble
		assert(norm(avgWeightGrad) < errorTol)
	}

	"posterior probabilitiy aggregates" should "be correclty calculated" in {
		val correctValue = {
			BDV.ones[Double](k) * 0.5 * nPoints.toDouble //posteriors should be of the form (0.5,...,0.5)
		}

		var error = (agg.posteriorsAgg - correctValue).t * (agg.posteriorsAgg - correctValue)

		assert(error < errorTol)
	}

	"outer product sum" should "be correclty calculated" in {
		val correctValue = points.map{case x => x * x.t * 0.5}.reduce(_ + _)
		
		val diff = outerProductsMats.map{case x => x - correctValue}.reduce(_ + _)
		
		assert(trace(diff*diff.t) < errorTol)

	}

	"gradient formula" should "return correct value" in {

		val correctValues = clusterDists.map{
			case dist =>  {
				//print(points(0) * points(0).t * 0.5 * nPoints.toDouble)
				0.5/nPoints.toDouble * (points(0) * points(0).t * 0.5 * nPoints.toDouble - dist.paramMat * 0.5 * nPoints.toDouble )
			}
		}

		val testValues = outerProductsMats.zip(clusterDists.zip(agg.posteriorsAgg.toArray)).map{
			case (o,(dist,p)) =>  {
				//print(o)
				0.5/nPoints.toDouble * (o - p*dist.paramMat)
			}
		}

		val totalError = correctValues.zip(testValues).map{case (x,y) => trace((x - y)*(x-y).t)}
		
		assert(totalError.sum < errorTol)

	}

}