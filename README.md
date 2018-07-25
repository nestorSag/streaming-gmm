# Gradient-based streamming Gaussian Mixtures in Spark
This project forms part of an MSc dissertation at the University of Edinburgh. 
It is based on the results from [1] about a stochastic gradient descent algorithm for (possibly regularized GMMs) and extends these results by implementing accelerated descent (actually, ascent) directions as well. Since a gradient-based mixture model can be easily extended to process streaming data in an intuitive way, such an implementation is also available.


[1] Hosseini, Reshad & Sra, Suvrit. (2017). An Alternative to EM for Gaussian Mixture Models: Batch and Stochastic Riemannian Optimization
