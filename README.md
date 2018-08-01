# Gradient-based streamming Gaussian Mixtures in Spark

This project forms part of an MSc dissertation at the University of Edinburgh. 

It is based on the results of [1] and implements stochastic gradient ascent and additional accelerated gradient ascent algorithms for GMM, which make it particularly fitting for large-scale or streaming mixture models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

The projectu uses ```scala 2.11.8```, ```spark 2.3.1``` and ```breeze 0.13.2```

### Installing

To use it as a dependency for other project you can add the following lines to your ```build.sbt``` file

```
lazy val projectIDependOn = RootProject(uri("git://github.com/nestorSag/streaming-gmm#v<latest_version>"))

dependsOn(projectIDependOn)
```

where ```latest_version``` is the latest release (as of now, ```1.3```)

## Using the program

Below is a summary of how the program works. To see the full documentation click [here](https://nestorsag.github.io/streaming-gmm/index.html#package)

### Initialization

The model is initialized using its factory object, either specifying the arrays of initial weights, mean vectors and covariance matrices:

```
import com.github.gradientgmm.models.GradientGaussianMixture

val model = GradientGaussianMixture(weigths,means,covs)
```

(here ```means``` and ```covs``` are Spark's ```DenseVector``` and ```DenseMatrix``` respectively)

or providing training data and the number of components (the data must be an ```RDD``` of Spark's ```DenseVector```, just like Spark's [GaussianMixtureModel](https://spark.apache.org/docs/2.3.1/api/scala/index.html#org.apache.spark.mllib.clustering.GaussianMixtureModel)):

```
val model = GradientGaussianMixture.init(data,k)
```

the above will use the result of a K-means model fitted with a small sample to set initial
weights, means and covariances.

We can initialize the model as above and then perform gradient ascent (actually, ascent) iterations in a single instruction with:

```
val model = GradientGaussianMixture.fit(data,k)
```

### Optimization

For an existing model, ```model.step(data)``` can be used to update it. The mini-batch size and number of iterations of every ```step()``` call can be specified beforehand:

```
model
.maxIter(20)
.batchSize(50)
.step(data)
```

### Optimization algorithms

The default optimization algorithm when creating and updating the model is ```GradientAscent```. Accelerated gradient ascent directions are also available (in fact they usually perform better, so we recommend using them) and we can create them as follows:

```
import com.github.gradientgmm.optim.algorithms.{MomentumGradientAscent,NesterovGradientAscent}

val myOptim = new MomentumGradientAscent()
    .setLearningRate(0.9)
    .halveStepSizeEvery(50)
    .setBeta(0.6) //inertia parameter
    .setMinLearningRate(0.01)

```
Now we can pass it to the model when initializing it by addind an ```optim``` parameter ton any of the instructions above, for example:

```
val model = GradientGaussianMixture.fit(data,k,myOptim)
```

### Regularization

To avoid the problem of [covariance singularity](https://stats.stackexchange.com/a/219358/66574), two regularization terms can be used; they are added to the optimizer object.

```
import com.github.gradientgmm.optim.regularization.{LogBarrier,ConjugatePrior}

val cpReg = new ConjugatePrior(k,dim) //pass k and data dimensionality
val lbReg = new LogBarrier()

optim.setRegularizer(lbReg)
model.setOptimizer(optim)
```
We recommend using ```LogBarrier``` because it is cheaper computationally and memory-wise and has a smaller effect on the final model's quality.

### Classifying Data

The methods ```predict``` and ```predictSoft``` do exactly the same as in Spark's [GaussianMixtureModel](https://spark.apache.org/docs/2.3.1/api/scala/index.html#org.apache.spark.mllib.clustering.GaussianMixtureModel)

### Streaming data

The methods ```predict```, ```predictSoft``` and ```step``` can also take a ```DStream``` object as input

## Authors

* **Nestor Sanchez - nestor.sag@gmail.com**

  **Amy Krause** 

## References
[1] Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization
