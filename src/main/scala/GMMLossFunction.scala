
class GMMLossFunction(
  val regularizer: Option[GMMLossTerm],
  var loss: Double,
  val weightGrads: Array[Double],
  val meanGrads: Array[Vector],
  val sigmaGrads: Array[Matrix]){

  private class GConcaveLoss() extends GMMLossTerm {

    def updateWeightGrad(
      currentWeight: Double, 
      posteriorProb: Double) = { posteriorProb - currentWeight }

    def updateMeanGrad(
      x: BDV, 
      mean: BDV, 
      posteriorProb: Double) = { posteriorProb*(x - mean)}

    def updateWeightGrad(
      x: Vector, 
      sigma: Matrix, 
      posteriorProb: Double) = {-BLAS.syr(-posteriorProb, x, sigma)}
  }


  val mainTerm = new GConcaveLoss()
  val k = meanGrads.length

  def updateWeightGrad(currentWeight: Double, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.updateWeightGrad(currentWeight,posteriorProb)

    case _ => mainTerm.updateWeightGrad(currentWeight,posteriorProb) + 
              regularizer.get.updateWeightGrad(currentWeight,posteriorProb)
  }

  def updateMeanGrad(x: Vector, mean:Vector, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.updateMeanGrad(x,mean,posteriorProb)

    case _ => mainTerm.updateMeanGrad(x,mean,posteriorProb) + 
              regularizer.get.updateMeanGrad(x,mean,posteriorProb)
  }

  def updateSigmaGrad(x: Vector, sigma:Vector, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.updateSigmaGrad(x,sigma,posteriorProb)

    case _ => mainTerm.updateSigmaGrad(x,sigma,posteriorProb) + 
              regularizer.get.updateSigmaGrad(x,sigma,posteriorProb)
  }


  def +=(x: GMMLossFunction): GMMLossFunction = {
    var i = 0
    while (i < k) {
      weightGrads(i) += x.weightGrads(i)
      meanGrads(i) += x.meanGrads(i)
      sigmaGrads(i) += x.sigmaGrads(i)
      i = i + 1
    }
    loss += x.loss
    this
  }


  def updateGrads(x: Vector,
                  weights: Array[Double], 
                  dists: Array[GConcaveMultivariateGaussian],  
                  posteriorProbs: Array[Double]): Unit = {
    
    for(j <- 0 to k-1){
      weightGrads(j) += updateWeightGrad(weights(j),posteriorProbs(j))
      meanGrads(j) += updateMeanGrad(x,dists(j).mean,posteriorProbs(j))
      sigmaGrads(j) += updateSigmaGrad(x,dists(j).sigma,posteriorProbs(j))
    }
  }

}

object GMMLossFunction { 

  def zero(regularizer: GMMLossTerm, k: Int, d: Int): GMMLossFunction = {
    new GMMLossFunction(
      Option(regularizer), 
      0.0, 
      Array.fill(k)(0.0),
      Array.fill(k)(BDV.zeros(d)), 
      Array.fill(k)(BreezeMatrix.zeros(d, d)))
  }

  def zero(k: Int, d: Int): GMMLossFunction = {
    new GMMLossFunction(
      None, 
      0.0, 
      Array.fill(k)(0.0),
      Array.fill(k)(BDV.zeros(d)), 
      Array.fill(k)(BreezeMatrix.zeros(d, d)))
  }

  // compute cluster contributions for each input point
  // (U, T) => U for aggregation
  def add(
      weights: Array[Double],
      dists: Array[GConcaveMultivariateGaussian])
      (lossFunc: GMMLossFunction, x: BV[Double]): GMMLossFunction = {
    val q = weights.zip(dists).map {
      case (weight, dist) => MLUtils.EPSILON + weight * dist.gConcavePdf(x) // <--only real change
    }
    val qSum = q.sum
    lossFunc.logLikelihood += math.log(pSum)
    lossFunc.updateGrad(x,weights,dists,q.map(x => x/qSum))

    lossFunc
  }
}


abstract class GMMLossTerm ={

  def updateWeightGrad(currentWeight: Double, posteriorProb: Double): Double

  def updateMeanGrad(x: Vector, mean:Vector, posteriorProb: Double): Vector

  def updateSigmaGrad(x: Vector, sigma:Matrix, posteriorProb: Double): Matrix

}