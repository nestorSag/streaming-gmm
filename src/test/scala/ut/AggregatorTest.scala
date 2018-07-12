import org.scalatest.FlatSpec


import com.github.nestorsag.gradientgmm.{GMMGradientAscent,GradientAggregator,UpdatableGaussianMixtureComponent}
import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, norm}


class AggregatorTest extends FlatSpec{
	
	
	var dim = 2
	var nPoints = 5
	var errorTol = 1e-8
	
	val clusterMeans = Array(new BDV(Array(-1.0,0.0)), new BDV(Array(1.0,0.0)))
	val clusterWeights = Array(0.5,0.5)
	val clusterVars = Array.fill(2)(BDM.eye[Double](dim))

	//val mixture = GradientBasedGaussianMixture(clusterWeights,clusterDists)
	val clusterDists = clusterMeans.zip(clusterVars).map{ case(m,v) => UpdatableGaussianMixtureComponent(m,v)}

	val optim = new GMMGradientAscent()
		.setLearningRate(0.5)
		.setShrinkageRate(1.0)

	val targetPoint = new BDV(Array(0.0,1.0))
	// do y = [x 1]
	val points = Array.fill(nPoints)(targetPoint).map{case v => new BDV(v.toArray ++ Array(1.0))}


	val adder = GradientAggregator.add(clusterWeights, clusterDists, optim, nPoints)_
	
	val agg = points.foldLeft(GradientAggregator.init(2,dim)){case (agg,point) => adder(agg,point)}

	"the log-likelihood" should "be correclty calculated" in {
		val correctValue = (-1.0 -math.log(2*math.Pi))

		var error = math.pow(agg.qLoglikelihood - correctValue,2)
		assert(error < errorTol)
	}

	"the posterior membership probabilities" should "be correclty calculated" in {
		val correctValue = Array.fill(2)(nPoints.toDouble/2)

		var error = new BDV(agg.weightsGradient) - new BDV(correctValue)
		assert(norm(error) < errorTol)
	}

	"the descent direction" should "be correclty calculated" in {
		val correctValue = {
			val v = new BDV(targetPoint.toArray ++ Array(1.0))

			clusterDists.map{ case d => (v*v.t - d.paramMat) * 0.25}
		}

		var error = correctValue.zip(agg.gaussianGradients).map{case (a,b) => trace((a-b).t*(a-b))}.sum
		assert(error < errorTol)
	}
}