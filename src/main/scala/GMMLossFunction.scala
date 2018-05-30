
abstract class GMMLossTerm extends Serializable ={

  def computeWeightGrad(currentWeight: Double, posteriorProb: Double): Double

  def computeMeanGrad(x: Vector, mean:Vector, posteriorProb: Double): Vector

  def computeSigmaGrad(x: Vector, sigma:Matrix, posteriorProb: Double): Matrix


}

abstract class GMMRegularizationTerm extends GMMLossTerm ={

  def computeLogPenaltyLoss(weights: Array[Double], 
                            dists: Array[GConcaveMultivariateGaussian]): Double


}


class GMMLossFunction(
  val regularizer: Option[GMMRegularizationTerm],
  var loss: Double,
  val weightGrads: Array[Double],
  val meanGrads: Array[Vector],
  val sigmaGrads: Array[Matrix]) extends Serializable{

  val k = meanGrads.length

  private class GConcaveLoss() extends GMMLossTerm {

    def computeWeightGrad(
      currentWeight: Double, 
      posteriorProb: Double) = { posteriorProb - currentWeight }

    def computeMeanGrad(
      x: BDV, 
      mean: BDV, 
      posteriorProb: Double) = { posteriorProb*(x - mean)}

    def computeSigmaGrad(
      x: Vector, 
      sigma: Matrix, 
      posteriorProb: Double) = {-BLAS.syr(-posteriorProb, x, sigma)}

    // def computeLogLoss(
    //   x: Vector,
    //   weights: Array[Double],
    //   dists: Array[GConcaveMultivariateGaussian]): Double = {

    //   val q = getPosteriorProbs(x,weights,dists)

    //   val qSum = q.sum

    //   math.log(qSum)
    // }

    // def getPosteriorProbs(
    //   x: Vector,
    //   weights: Array[Double],
    //   dists: Array[GConcaveMultivariateGaussian]): Array[Double] = {

    //   weights.zip(dists).map {
    //     case (weight, dist) => MLUtils.EPSILON + weight * dist.gConcavePdf(x) // <--only real change
    //   }

    // }

  }


  val mainTerm = new GConcaveLoss()

  def computeWeightGrad(currentWeight: Double, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.computeWeightGrad(currentWeight,posteriorProb)

    case _ => mainTerm.computeWeightGrad(currentWeight,posteriorProb) + 
              regularizer.get.computeWeightGrad(currentWeight,posteriorProb)
  }

  def computeMeanGrad(x: Vector, mean:Vector, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.computeMeanGrad(x,mean,posteriorProb)

    case _ => mainTerm.computeMeanGrad(x,mean,posteriorProb) + 
              regularizer.get.computeMeanGrad(x,mean,posteriorProb)
  }

  def computeSigmaGrad(x: Vector, sigma:Vector, posteriorProb: Double): Double = regularizer match {

    case None => mainTerm.updateSigmaGrad(x,sigma,posteriorProb)

    case _ => mainTerm.computeSigmaGrad(x,sigma,posteriorProb) + 
              regularizer.get.computeSigmaGrad(x,sigma,posteriorProb)
  }

  def computeRegularizationLoss(weights: Array[Double], 
                           dists: Array[GConcaveMultivariateGaussian]): Double = regularizer match{

    case None => 0
    case _ => regularizer.get.computeLogPenaltyLoss(weights,dists)
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
      weightGrads(j) += computeWeightGrad(weights(j),posteriorProbs(j))
      meanGrads(j) += computeMeanGrad(x,dists(j).mean,posteriorProbs(j))
      sigmaGrads(j) += computeSigmaGrad(x,dists(j).sigma,posteriorProbs(j))
    }
  }

}

object GMMLossFunction extends Serializable{ 

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
      case (weight, dist) => MLUtils.EPSILON + weight * dist.gConcavePdf(x)
    }
    val qSum = q.sum

    lossFunc.loss += math.log(qSum) + lossFunc.computeRegularizationLoss(weights,dists)
    lossFunc.updateGrad(x,weights,dists,q.map(x => x/qSum))
    lossFunc

  }
}

