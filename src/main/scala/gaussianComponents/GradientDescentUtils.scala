package streamingGmm

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}

trait GradientDescentUtils extends Serializable{

  val d: Int 

  private[streamingGmm] var momentum: Option[BDM[Double]] = None

  private[streamingGmm] var adamInfo: Option[BDM[Double]] = None //raw second moment gradient estimate (for Adam optimizer)

  private[streamingGmm] def initializeMomentum: Unit = {
    momentum = Option(BDM.zeros[Double](d+1,d+1))
  }

  def removeMomentum: Unit = {
    momentum = None
  }

  private[streamingGmm] def updateMomentum(mat: BDM[Double]): Unit = {
    momentum = Option(mat)
  }

  private[streamingGmm] def initializeAdamInfo: Unit = {
    adamInfo = Option(BDM.zeros[Double](d+1,d+1))
  }

  def removeAdamInfo: Unit = {
    adamInfo = None
  }

  private[streamingGmm] def updateAdamInfo(mat: BDM[Double]): Unit = {
    adamInfo = Option(mat)
  }

  def update(newParamsMat: BDM[Double]): Unit
}