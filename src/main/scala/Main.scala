import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}



object Main extends App {
 	// Loads data
 	val conf = new SparkConf()
 		.setAppName("streamming-gmm-test")
 		.setMaster("local")

 	val sc = new SparkContext(conf)

	val data = sc.textFile("src/test/resources/testdata.csv")// Trains Gaussian Mixture Model
	val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

	val gmm = new GaussianMixture().setK(4).run(parsedData)

	// output parameters of mixture model model
	for (i <- 0 until gmm.k) {
	      println("weight=%f\nmu=%s\nsigma=\n%s\n" format
	        (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
	    }

	sc.stop()
}


