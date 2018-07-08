package net.github.gradientgmm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.rdd.RDD

trait GMMOptimizer extends Serializable{

	private[gradientgmm] val regularizer: Option[GMMRegularizer] = None
	private[gradientgmm] val weightOptimizer: GMMWeightTransformation = new SoftmaxWeightTransformation()
	private[gradientgmm] var learningRate: Double = 0.9
	private[gradientgmm] var minLearningRate: Double = 1e-2
	private[gradientgmm] var shrinkageRate: Double = 0.95
	private[gradientgmm] var batchSize: Option[Int] = None
	private[gradientgmm] var convergenceTol: Double = 1e-6
	private[gradientgmm] var maxIter: Int = 100

	def setLearningRate(learningRate: Double): this.type = { 
		require(learningRate > 0 , "learning rate must be positive")
		this.learningRate = learningRate
		this
	}

	def getLearningRate: Double = { 
		this.learningRate
	}

	def setMinLearningRate(m: Double): this.type = { 
		require(m >= 0, "minLearningRate rate must be in positive")
		minLearningRate = m
		this
	}

	def getMinLearningRate: Double = { 
		minLearningRate
	}

	def setShrinkageRate(s: Double): this.type = { 
		require(s > 0 &&  s <= 1.0, "learning rate must be in (0,1]")
		shrinkageRate = s
		this
	}

	def getBatchSize: Option[Int] = batchSize

	def setBatchSize(n: Int): this.type = {
		require(n>0,"n must be a positive integer")
		batchSize = Option(n)
		this
	}

	def getShrinkageRate: Double = { 
		shrinkageRate
	}

	def updateLearningRate: Unit = {
		learningRate = math.max(shrinkageRate*learningRate,minLearningRate)
	}

	def fromSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.fromSimplex(weights)
	}

	def toSimplex(weights: BDV[Double]): BDV[Double] = {
		weightOptimizer.toSimplex(weights)
	}

	def getConvergenceTol: Double = convergenceTol

	def setConvergenceTol(x: Double): this.type = {
		require(x>0,"convergenceTol must be positive")
		convergenceTol = x
		this
	}


	def setMaxIter(m: Int): this.type = {
		require(m > 0 ,s"maxIter needs to be a positive integer; got ${m}")
		maxIter = m
		this
	}

	def getMaxIter: Int = {
		maxIter
	}

	private[gradientgmm] def lossGradient(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double] = {

		regularizer match{
			case None => basicLossGradient(dist.paramMat,point,w) 
			case Some(_) => basicLossGradient(dist.paramMat,point,w) +
				regularizer.get.gradient(dist)
		}

	}

	private[gradientgmm] def basicLossGradient(paramMat: BDM[Double], point: BDM[Double], w: Double): BDM[Double] = {

		(point - paramMat) * 0.5 * w
	}


	private[gradientgmm] def basicSoftWeightsGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		weightOptimizer.gradient(posteriors,weights)
	}

	private[gradientgmm] def softWeightGradient(posteriors: BDV[Double], weights: BDV[Double]): BDV[Double] = {

		var grads = regularizer match {
			case None => basicSoftWeightsGradient(posteriors,weights)
			case Some(_) => basicSoftWeightsGradient(posteriors,weights) +
		 			regularizer.get.softWeightsGradient(weights)
		}

		grads(weights.length - 1) = 0.0

		grads

	}

	def evaluateRegularizationTerm(dist: UpdatableGConcaveGaussian,weight: Double): Double = {

		regularizer match{
			case None => 0
			case Some(_) => regularizer.get.evaluate(dist,weight)
		}

	}

	def fit(data: RDD[SV], k: Int = 2, startingSampleSize: Int = 50, kMeansIters: Int = 20, seed: Int = 0): GradientBasedGaussianMixture = {
		
		val model = GradientBasedGaussianMixture.initialize(
			data,
			this,
			k,
			startingSampleSize,
			kMeansIters,
			seed)
		    
		model.step(data)

		model
	}

	def direction(dist: UpdatableGConcaveGaussian, point: BDM[Double], w: Double): BDM[Double]

	def softWeightsDirection(posteriors: BDV[Double], weights: WeightsWrapper): BDV[Double]


}