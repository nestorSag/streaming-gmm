import org.scalatest.FlatSpec


import com.github.gradientgmm.components.UpdatableWeights
import com.github.gradientgmm.optim.algorithms.ParameterOperations

import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace, sum}
import breeze.numerics.sqrt


trait OptimTestSpec extends FlatSpec{
	
	
	var dim = 2
	var niter = 5
	var errorTol = 1e-8
	var k = 5
	//get random matrix

	val randA = BDM.rand(dim,dim)
	val tcov = randA.t * randA + BDM.eye[Double](dim) // form SPD covariance matrix

	//get random mean vector 

	val mu = BDV.rand(dim)

	val targetParamMat: BDM[Double] = {
    // build target parameter matrix
	    val lastRow = new BDV[Double](mu.toArray ++ Array[Double](1))

	    BDM.vertcat(BDM.horzcat(tcov + mu*mu.t,mu.asDenseMatrix.t),lastRow.asDenseMatrix)

  	}

  	def toBDV(x: Array[Double]): BDV[Double] = {
  		new BDV(x)
  	}

  	val targetWeights = BDV.rand(k)
	targetWeights /= sum(targetWeights)

	val initialWeights = (1 to k).map{ case x => 1.0/k}.toArray

	var weightObj = new UpdatableWeights(initialWeights)





	val vectorOps = new ParameterOperations[BDV[Double]] {

      def sum(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x + y}
      def sumScalar(x: BDV[Double], z: Double): BDV[Double] = {x + z}
      def rescale(x: BDV[Double], z: Double): BDV[Double] = {x*z}
      def sub(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x - y}

      def ewProd(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x *:* y}
      def ewDiv(x: BDV[Double], y: BDV[Double]): BDV[Double] = {x /:/ y}
      def ewSqrt(x:BDV[Double]): BDV[Double] = {sqrt(x)}
  }

  val matrixOps = new ParameterOperations[BDM[Double]] {

      def sum(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x + y}
      def sumScalar(x: BDM[Double], z: Double): BDM[Double] = {x + z}
      def rescale(x: BDM[Double], z: Double): BDM[Double] = {x*z}
      def sub(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x - y}

      def ewProd(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x *:* y}
      def ewDiv(x: BDM[Double], y: BDM[Double]): BDM[Double] = {x /:/ y}
      def ewSqrt(x:BDM[Double]): BDM[Double] = {sqrt(x)}
  }


}