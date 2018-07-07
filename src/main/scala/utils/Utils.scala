package net.github.gradientgmm

import breeze.linalg.{DenseVector => BDV}

private[gradientgmm] object Utils extends Serializable{

  val simplexErrorTol = 1e-8

  val EPS = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  def toBDV(x: Array[Double]): BDV[Double] = {
    new BDV(x)
  }

}