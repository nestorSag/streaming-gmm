library(plotly)
library(ggplot2)
source("RUtils/utils.R")

testData = read.table(testDataPath,sep=" ",header=FALSE)
eps = 0.1 #put the trajectory a bit above the simplex for visualisation purposes

#true parameter values
trueCenters = list(center1,as.numeric(rotation%*%center1),as.numeric(rotation%*%rotation%*%center1))

trueWeights = (1+eps)*c(500,250,125)/nrow(testData)

trueWeightsDf = data.frame(w1=trueWeights[1],w2=trueWeights[2],w3=trueWeights[3])

#weights can be in any order

l = logParser(logsPath)

saveMeansPlot(l$means,trueCenters)

g = displayWeightsPlot(l$weights,trueWeightsDf)
htmlwidgets::saveWidget(as_widget(g), "weights.html")
system("mv weights.html RUtils/pathPlots/weights.html") ##to deal with a bug in htmlwidgets::saveWidget

saveLLPlot(l$ll)