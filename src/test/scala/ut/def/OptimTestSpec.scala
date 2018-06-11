import org.scalatest.FlatSpec
import breeze.linalg.{diag, eigSym, max, DenseMatrix => BDM, DenseVector => BDV, Vector => BV, trace}


trait OptimTestSpec extends FlatSpec{
	
	
	var dim = 5
	var niter = 5
	var errorTol = 1e-8
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

}