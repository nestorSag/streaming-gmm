class GConcaveLoss private (val prior: Option[Prior]){
 	
	def q_likelihood(y: Vector, gaussians: List[GConcaveGaussian], weights: List[Double]) = {

		val f = weights.zip(gaussians).map {
			case (w,g) => w * g.q(y) //q(y) = math.sqrt(2*math.Pi) * math.exp(0.5) * g.gConcavePdf(y)
		}

		f.sum
	}

	def evaluate(data: RDD[Vector], gaussians: List[GConcaveGaussian], weights: List[Double]): Double = {

		val regularization: Double = prior match {
			Some => prior.evaluate(gaussians,weight)
			None => 0.0
		}

		regularization + evaluateLikelihood(data,gaussians,weights)
	}

	def evaluateLikelihood(data: RDD[Vector], gaussians: List[GConcaveGaussian], weights: List[Double]): Double = {
		data.map(x => math.log(q_likelihood(x++1,gaussians,weights))).sum
	}

	def getGradient(data: RDD[Vector], gaussians: List[GConcaveGaussian], weights: List[Double]): Double = {

		if(prior.isEmpty()){
			getLikelihoodGradient(data,gaussians,weights)	
		}else{
			getLikelihoodGradient(data,gaussians,weights).zip(prior.getGradient(gaussians,weight)).map{((gS1,gw1),(gS2,gw2)) => (gS1 + gS2, gw1 + gw2)}
		}

	}

	def getLikelihoodGradient(data: RDD[Vector], gaussians: List[GConcaveGaussian], weights: List[Double]): List((Matrix,Double)) = {

		val data = data.map(x => (x++1).toBreeze)

		val gradients = weights.zip(gaussians).map{
			case (w,dist) => (w,dist,dist.gConcaveSigma()) //get respective weight and s_j variance matrix
			}.map{
				case (w,dist,s) => {

					//responsability of k-th cluster for this datapoint
					//it is also the posterior probability that this data point comes from the k-th cluster
					val responsability = data.map{y => w*dist.q(y)/q_likelihood(y,gaussians,weights)} 

					//grad of matrix S_j which wraps both cov matrix and mean vector
					val grad_s = responsability.zip(data).foldLeft(0*s){(total,(resp,y)) => total + 0.5*resp*(y*y.t - s)}
					
					//grad of log weight ratios
					val grad_w = responsability.foldLeft(0){(total,resp) => total + (resp - w)}
					(grad_s,grad_w)
				}
			}
	}

}
