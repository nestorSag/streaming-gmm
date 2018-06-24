import org.scalatest.FlatSpec


import streamingGmm.SGDWeights
import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}


trait OptimTestSpec extends FlatSpec{
	
	
	var dim = 10
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
	targetWeights /= targetWeights.toArray.sum

	val initialWeights = (1 to k).map{ case x => 1.0/k}

	var weightObj = new SGDWeights(initialWeights.toArray)


}