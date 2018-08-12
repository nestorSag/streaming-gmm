package com.github.gradientgmm

import com.github.gradientgmm.components.{UpdatableGaussianComponent, UpdatableWeights, Utils}
import com.github.gradientgmm.optim.{Optimizable, Optimizer, GradientAscent}

import breeze.linalg.{diag, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.mllib.linalg.{Matrix => SM, Vector => SV, Vectors => SVS, Matrices => SMS}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel, GaussianMixtureModel}
import org.apache.spark.mllib.stat.distribution.{MultivariateGaussian => SMG}


import org.apache.log4j.Logger

/**
  * Optimizable gradient-based Gaussian Mixture Model
  * See [[https://arxiv.org/abs/1706.03267 An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization'']]
  * @param w Weight vector wrapper
  * @param g Array of mixture components (distributions)
  * @param optim Optimization object
 
  */
class GradientGaussianMixture private (
  w:  UpdatableWeights,
  g: Array[UpdatableGaussianComponent],
  var optim: Optimizer) extends UpdatableGaussianMixture(w,g) with Optimizable {


/**
  * mini-batch size as fraction of the complete training data
  * Spark needs a double in [0,1] to take samples, so we need
  * to translate a given batch size to a fraction of the data size
  */
  var batchFraction = 1.0


/**
  * Optimize the mixture parameters given some training data
  * @param data Training data as an RDD of Spark vectors 
 
  */
  def step(data: RDD[SV]): this.type = {

    // initialize logger. It logs the parameters' paths to solution
    // the messages' leve; lis set to DEBUG, so be sure to set the log level to DEBUG 
    // if you want to see them
    val logger: Logger = Logger.getLogger("modelPath")

    val d = data.first().size

    val sc = data.sparkContext

    //map original vectors to points for the g-concave formulation
    // y = [x 1]
    val gConcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))}.cache()

    val shouldDistribute = shouldDistributeGaussians(k, d)

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0
    
    // broadcast optim and reg to workers
    val bcOptim = sc.broadcast(this.optim)
    // broadcast optim and reg to workers
    val bcReg = sc.broadcast(this.regularizer)

    val initialRate = optim.getLearningRate



    val(ebs,fraction): (Double,Double) = { //ebs = epected batch size
      val dataSize = gConcaveData.count()

      if(batchSize.isDefined){
        // this is to prevent that 0 size samples are too frequent
        // this is because of the way spark takes random samples
        // we want Prob(0 size sample) <= 1e-3
        val safeBatchSize: Double = dataSize*(1 - math.exp(math.log(1e-3)/dataSize))
        val correctedBatchSize = math.max(batchSize.get.toDouble,safeBatchSize)
        (correctedBatchSize,correctedBatchSize/dataSize)
      }else{
        (dataSize,1.0)
      }
    }
    logger.debug(s"expected batch size: ${ebs}, fraction: ${fraction}")
    batchFraction = math.min(fraction,1.0) //in case batches are larger than whole dataset
    
    //a bit of syntactic sugar
    def toSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.toSimplex
    def fromSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.fromSimplex

    //sampling seed

    while (iter < maxIter && math.abs(newLL-oldLL) > convergenceTol) {

      globalIterCounter += 1 //this is to avoid taking the same sample each iteration

      val t0 = System.nanoTime //this is to time program

      // if model parameters can be plotted (specific d and k)
      // and logger is set to debug, send trajectory of estimators to logs
      if(d==2 && k == 3){
        //send values formatted for R processing to logs
        logger.debug(s"means: list(${gaussians.map{case g => "c(" + g.getMu.toArray.mkString(",") + ")"}.mkString(",")})")
        logger.debug(s"weights: ${"c(" + weights.weights.mkString(",") + ")"}")
        logger.debug(s"covs: list(${gaussians.map{case g => "c(" + g.getSigma.toArray.mkString(",") + ")"}.mkString(",")})")
      }

      // initialize curried adder that will aggregate the necessary statistics in the workers
      val adder = sc.broadcast(
        MetricAggregator.add(weights.weights, gaussians)_)

      val sampleStats = batch(gConcaveData).treeAggregate(MetricAggregator.init(k, d))(adder.value, _ += _)

      val n: Int = sampleStats.counter // number of actual data points in current batch

      if(n>0){
      // pair Gaussian components with their respective parameter gradients
        val tuples =
            Seq.tabulate(k)(i => (
              sampleStats.outerProductsAgg(i),
              sampleStats.posteriorsAgg(i),
              gaussians(i),
              n.toDouble))

        // update gaussians
        val (newDists, regValues) = if (shouldDistribute) {
          // compute new gaussian parameters and regularization values in
          // parallel

          val numPartitions = math.min(k, 1024) // same as GaussianMixture (MLlib)

          val (newDists,regValue) = sc.parallelize(tuples, numPartitions).map { case (outer,w,dist,_n) =>

            val _Y = completeMatrix(outer)

            //gradient for Gaussian parameters
            val (grad, regValue) = if(bcReg.value.isDefined){
              (((_Y - w * dist.paramMat) * 0.5 + bcReg.value.get.gaussianGradient(dist)) / _n,
                bcReg.value.get.evaluateDist(dist)/_n)
            }else{
              (((_Y - w * dist.paramMat) * 0.5 ) / _n, 0.0)
            }

            val newPars = bcOptim.value.getUpdate(
                dist.paramMat,
                grad,
                dist.optimUtils)
            
            dist.update(newPars)

            bcOptim.value.updateLearningRate //update learning rate in workers

            (dist, regValue)

          }.collect().unzip

          (newDists.toArray,regValue.toArray)

        } else {

          val (newDists,regValue) = tuples.map { case (outer,w,dist,_n) =>

            val _Y = completeMatrix(outer)

            //gradient for Gaussian parameters
            val (grad, regValue) = if(regularizer.isDefined){
              (((_Y - w * dist.paramMat) * 0.5 + regularizer.get.gaussianGradient(dist)) / _n,
                regularizer.get.evaluateDist(dist)/_n)
            }else{
              (((_Y - w * dist.paramMat) * 0.5 ) / _n, 0.0)
            }

            dist.update(
              optim.getUpdate(
                dist.paramMat,
                grad, //averaged gradient. see line 136
                dist.optimUtils))
            
            (dist, regValue)

          }.unzip

          (newDists.toArray,regValue.toArray)

        }

        gaussians = newDists
        
        val breezeWeights = Utils.toBDV(weights.weights)

        val regWeightValue = if(regularizer.isDefined){
          regularizer.get.evaluateWeights(breezeWeights)/n.toDouble
        }else{
          0.0
        }

        val weightsGrads = if(regularizer.isDefined){
          (sampleStats.weightsGradient + regularizer.get.weightsGradient(breezeWeights)) / n.toDouble 
        }else{
          sampleStats.weightsGradient /n.toDouble
        }

       weightsGrads(weightsGrads.length - 1) = 0.0 // last weight's auxiliar variable is fixed because of the simplex cosntraint

        val newWeights = optim.getUpdate(
              fromSimplex(breezeWeights),
              weightsGrads, 
              weights.optimUtils)

        weights.update(toSimplex(newWeights))

        oldLL = newLL // current becomes previous

        newLL = (sampleStats.loss + regValues.sum + regWeightValue) / n.toDouble //average loss

        optim.updateLearningRate //update learning rate in driver
        iter += 1

        val elapsed = (System.nanoTime - t0)/1e9d
        logger.info(s"iteration ${iter} took ${elapsed} seconds for ${n} samples. new LL: ${newLL}")
        
      }else{
        logger.info("No points in sample. Skipping iteration")
      }

      //adder.unpersist()
    }

    //bcOptim.destroy()

    //set learning rate to original value in case it was shrunk
    optim.setLearningRate(initialRate)

    this

  }


/**
  * Optimize the mixture parameters given some training data
  * @param data Training data as an Array of Breeze vectors 
 
  */

  def step(data: Array[BDV[Double]]): this.type = {

    require(batchSize.isDefined,"batch size is not set")
    // initialize logger. It logs the parameters' paths to solution
    // the messages' leve; lis set to DEBUG, so be sure to set the log level to DEBUG 
    // if you want to see them
    val logger: Logger = Logger.getLogger("modelPath")

    val d = data(0).size

    //map original vectors to points for the g-concave formulation
    // y = [x 1]
    val gConcaveData = data.map{x => new BDV[Double](x.toArray ++ Array[Double](1.0))}
    val N = gConcaveData.length

    var newLL = 1.0   // current log-likelihood
    var oldLL = 0.0  // previous log-likelihood
    var iter = 0

    val initialRate = optim.getLearningRate

    val batchLength = if(batchSize.isDefined){
      batchSize.get
    }else{
      gConcaveData.length
    }

    val epochs = math.ceil(batchLength * maxIter.toDouble / N)
    var epoch = 0


    val batchesPerEpoch = math.floor(N.toDouble/batchLength)

    val batchData = gConcaveData.sliding(batchLength)

    while (epoch < epoch) {

      var batches = math.floor((batchLength * maxIter.toDouble - epoch * N) / batchLength)
      var batch = 0

      while(batch < batches){
      
        val t0 = System.nanoTime //this is to time program

        // initialize curried adder that will aggregate the necessary statistics in the workers
        val discard = (batchesPerEpoch - batches).toInt

        if(discard > 1){
          batchData.drop(discard-1).foldLeft(this){case (model,batch) => model._step(batch)}
        }else{
          batchData.foldLeft(this){case (model,batch) => model._step(batch)}
        }

        val elapsed = (System.nanoTime - t0)/1e9d
        logger.info(s"iteration took ${elapsed}}")

        batch += 1
      }

      epoch += 1
    }

    //set learning rate to original value in case it was shrunk
    optim.setLearningRate(initialRate)

    this
  }

  def _step(batch: Array[BDV[Double]]): this.type = {

    //a bit of syntactic sugar
    def toSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.toSimplex
    def fromSimplex: BDV[Double] => BDV[Double] = optim.weightsOptimizer.fromSimplex

    val k = weights.length
    val d = batch(0).length

     val adder = MetricAggregator.add(weights.weights, gaussians)_

    val sampleStats = batch
      .foldLeft(MetricAggregator.init(k,d)){case (agg,point) => adder(agg,point)}
    
    val n = sampleStats.counter

    val tuples =
      Seq.tabulate(k)(i => (
        sampleStats.outerProductsAgg(i),
        sampleStats.posteriorsAgg(i),
        gaussians(i),
        n.toDouble))

    val (newDists, regValues) = {

      val (newDists,regValue) = tuples.map { case (outer,w,dist,_n) =>

          val _Y = completeMatrix(outer)

          //gradient for Gaussian parameters
          val (grad, regValue) = if(regularizer.isDefined){
            (((_Y - w * dist.paramMat) * 0.5 + regularizer.get.gaussianGradient(dist)) / _n,
              regularizer.get.evaluateDist(dist)/_n)
          }else{
            (((_Y - w * dist.paramMat) * 0.5 ) / _n, 0.0)
          }

          dist.update(
            optim.getUpdate(
              dist.paramMat,
              grad, //averaged gradient. see line 136
              dist.optimUtils))
          
          (dist, regValue)

        }.unzip

        (newDists.toArray,regValue.toArray)
    }

    gaussians = newDists
      
    val breezeWeights = Utils.toBDV(weights.weights)

    val regWeightValue = if(regularizer.isDefined){
      regularizer.get.evaluateWeights(breezeWeights)/n.toDouble
    }else{
      0.0
    }

    val weightsGrads = if(regularizer.isDefined){
      (sampleStats.weightsGradient + regularizer.get.weightsGradient(breezeWeights)) / n.toDouble 
    }else{
      sampleStats.weightsGradient /n.toDouble
    }

   weightsGrads(weightsGrads.length - 1) = 0.0 // last weight's auxiliar variable is fixed because of the simplex cosntraint

    val newWeights = optim.getUpdate(
          fromSimplex(breezeWeights),
          weightsGrads, 
          weights.optimUtils)

    weights.update(toSimplex(newWeights))

    // oldLL = newLL // current becomes previous

    // newLL = (sampleStats.loss + regValues.sum + regWeightValue) / n.toDouble //average loss

    this

  }

/**
  * Returns a Spark's Gaussian Mixture Model with the current parameters initialized with the current parameters
 
  */
  def toSparkGMM: GaussianMixtureModel = {

    val d = gaussians(0).getMu.length

    new GaussianMixtureModel(
      weights.weights,
      gaussians.map{
        case g => new SMG(
          SVS.dense(g.getMu.toArray),
          SMS.dense(d,d,g.getSigma.toArray))})
  }

/**
  * Update model parameters using streaming data
  * @param data Streaming data
 
  */
  def step(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      step(rdd)
    }
  }

/**
  * Cluster membership prediction for streaming data
  * @param data Streaming data
 
  */
  def predict(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predict(rdd)
    }
  }

/**
  * Soft cluster membership prediction for streaming data
  * @param data Streaming data
 
  */
  def predictSoft(data: DStream[SV]) {
    data.foreachRDD { (rdd, time) =>
      predictSoft(rdd)
    }
  }


/**
  * Heuristic to decide when to distribute the computations. Taken from Spark's GaussianMixture class
 
  */
  private def shouldDistributeGaussians(k: Int, d: Int): Boolean = ((k - 1.0) / k) * d > 25


/**
  * take sample for the current mini-batch, or pass the whole dataset if optim.batchSize = None
 
  */
  private def batch(data: RDD[BDV[Double]])(implicit globalIterCounter: Long): RDD[BDV[Double]] = {
    if(batchFraction < 1.0){
      data.sample(false,batchFraction,seed + globalIterCounter)
    }else{
      data
    }
  }

  /**
  * Build a symmetric matrix from an array that represents an upper triangular matrix
  *
  * @param x upper triangular matrix array
 
  */

  private def completeMatrix(x: Array[Double]): BDM[Double] = {

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

}

object GradientGaussianMixture{

/**
  * Creates a new GradientGaussianMixture instance from arrays of weights, means and covariances
  * @param weights Array of weights
  * @param means Array of mean vectors
  * @param covs Array of covariance matrices
  * @param optim Optimization algorithm. Defaults to GradintAscent
 
  */
  def apply(
    weights: Array[Double],
    means: Array[BDV[Double]],
    covs: Array[BDM[Double]],
    optim: Optimizer): GradientGaussianMixture = {

    new GradientGaussianMixture(
      new UpdatableWeights(weights),
      means.zip(covs).map{case(m,v) => UpdatableGaussianComponent(m,v)},
      optim)
  }

/**
  * Creates a new GradientGaussianMixture instance from arrays of weights, means and covariances
  * @param weights Array of weights
  * @param means Array of mean vectors
  * @param covs Array of covariance matrices
  * @param optim Optimization algorithm. Defaults to GradintAscent
 
  */
  def apply(
    weights: Array[Double],
    means: Array[SV],
    covs: Array[SM],
    optim: Optimizer = new GradientAscent()): GradientGaussianMixture = {

    val covdim = covs(0).numCols

    new GradientGaussianMixture(
      new UpdatableWeights(weights),
      means.zip(covs).map{case(m,v) => UpdatableGaussianComponent(Utils.toBDV(m.toArray),new BDM(covdim,covdim,v.toArray))},
      optim)
  }

/**
  * Creates a new GradientGaussianMixture instance initialized with the
  * results of a K-means model fitted with a sample of the data
  * @param data training data in the form of an RDD of Spark vectors
  * @param optim Optimizer object. Defaults to simple gradient ascent
  * @param k Number of components in the mixture
  * @param pointsPerCl The K-Means model will be trained with k*pointsPerCl points
  * @param nIters Number of iterations allowed for the K-means model
  * @param nTries Number of K-means models to try
  * @param seed random seed
  */
  def init(
    data: RDD[SV],
    k: Int,
    optim: Optimizer = new GradientAscent(),
    pointsPerCl: Int = 50,
    nIters: Int = 20,
    nTries: Int = 1,
    seed: Long = 0): GradientGaussianMixture = {
    
    val dataSize = data.count()

    val sc = data.sparkContext
    val d = data.take(1)(0).size //get data dimensionality
    val n = math.min(dataSize,pointsPerCl*k).toInt //in case the data has too few points
    var samples = sc.parallelize(data.takeSample(withReplacement = false, n, seed))

    var kmeansModel = new KMeans()
      .setMaxIterations(nIters)
      .setK(k)
      .setSeed(seed)
      .run(samples)

    // within-set-sum-of-squares-error
    var WSSSE: Double = kmeansModel.computeCost(samples)/n

    // select best model from many tries
    for(i <- 2 to nTries){

      val model = new KMeans()
      .setMaxIterations(nIters)
      .setK(k)
      .setSeed(seed + i - 1)
      .run(samples)

      var cost = model.computeCost(samples)/n

      if(cost < WSSSE){
        WSSSE = cost
        kmeansModel = model
      }

    }
    
    val means = kmeansModel.clusterCenters.map{case v => Utils.toBDV(v.toArray)}

    //add means to sample points to avoid having cluster with zero points 
    samples = samples.union(sc.parallelize(means.map{case v => SVS.dense(v.toArray)}))

    // broadcast values to compute sample covariance matrices
    val kmm = sc.broadcast(kmeansModel)
    val scMeans = sc.broadcast(means)

    // get empirical cluster proportions to initialize the mixture/s weights
    //add 1 to counts to avoid division by zero
    val proportions = samples
      .map{case s => (kmm.value.predict(s),1)}
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()
      .map{case (k,p) => p.toDouble}

    val scProportions = sc.broadcast(proportions)

    //get empirical covariance matrices
    //also add a rescaled identity matrix to avoid starting with singular matrices 
    val pseudoCov = samples
      .map{case v => {
          val prediction = kmm.value.predict(v)
          val denom = math.sqrt(scProportions.value(prediction))
          (prediction,(Utils.toBDV(v.toArray)-scMeans.value(prediction))/denom) }} // x => (x-mean)
      .map{case (k,v) => (k,v*v.t)}
      .reduceByKey(_ + _)
      .map{case (k,v) => {
        val avgVariance = math.max(1e-4,trace(v))/d
        (k,v + BDM.eye[Double](d) * avgVariance)
        }}
      .sortByKey()
      .collect()
      .map{case (k,m) => m}

    new GradientGaussianMixture(
      new UpdatableWeights(proportions.map{case p => p/proportions.sum}), 
      (0 to k-1).map{case i => UpdatableGaussianComponent(means(i),pseudoCov(i))}.toArray,
      optim)

  }

  /**
  * Fit a Gaussian Mixture Model (see [[https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model]]).
  * The model is initialized using a K-means algorithm over a small sample and then 
  * fitting the resulting parameters to the data using this {GMMOptimization} object
  * @param data Data to fit the model
  * @param optim Optimization algorithm
  * @param k Number of mixture components (clusters)
  * @param batchSize number of samples processed per iteration
  * @param maxIter maximum number of gradient ascent steps allowed
  * @param convTol log-likelihood change tolerance for stopping criteria
  * @param pointsPerCl The K-Means model will be trained with k*pointsPerCl points
  * @param kMeansIters Number of iterations allowed for the K-means algorithm
  * @param nTries Number of K-means models to try
  * @param seed Random seed
  * @return Fitted model
  */
  def fit(
    data: RDD[SV], 
    k: Int,
    optim: Optimizer = new GradientAscent(), 
    batchSize: Option[Int] = None,
    maxIter: Int = 100,
    convTol: Double = 1e-6, 
    pointsPerCl: Int = 50,
    kMeansIters: Int = 20,
    kMeansTries: Int = 1,
    seed: Int = 0): GradientGaussianMixture = {
    
    val model = init(
                  data,
                  k,
                  optim,
                  pointsPerCl,
                  kMeansIters,
                  kMeansTries,
                  seed)
    
    if(batchSize.isDefined){
      model.setBatchSize(batchSize.get)
    }

    model
    .setMaxIter(maxIter)
    .setConvergenceTol(convTol)
    .step(data)

    model
  }
}