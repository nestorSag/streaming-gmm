// package com.github.nestorsag.gradientgmm

// import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

// trait VectorGradientUtils extends Serializable{

//   val d: Int
//   var momentum: Option[BDV[Double]] = None
//   var adamInfo: Option[BDV[Double]] = None

//   private[gradientgmm] def updateMomentum(x: BDV[Double]): Unit = {
//     momentum = Option(x)
//   }

//   private[gradientgmm] def updateAdamInfo(x: BDV[Double]): Unit = {
//     adamInfo = Option(x)
//   }

//   private[gradientgmm] def initializeMomentum: Unit = {
//     momentum = Option(BDV.zeros[Double](d))
//   }

//   private[gradientgmm] def initializeAdamInfo: Unit = {
//      adamInfo = Option(BDV.zeros[Double](d))
//   }

//   private def removeMomentum: Unit = {
//     momentum = None
//   }

//   private def removeAdamInfo: Unit = {
//     adamInfo = None
//   }

//   def lighten: this.type = {
//     removeMomentum
//     removeAdamInfo
//     this
//   }

// }
