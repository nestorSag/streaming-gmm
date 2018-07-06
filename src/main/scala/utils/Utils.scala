package edu.github.gradientgmm

object Utils{
  private[gradientgmm] val EPS = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }
}