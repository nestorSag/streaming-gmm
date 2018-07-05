library(stringr)

rotation = matrix(c(cos(2*pi/3),sin(2*pi/3),-sin(2*pi/3),cos(2*pi/3)),2,2)
logsPath = "logs/reports.log"
testDataPath = "src/test/resources/testdata.csv"

#center of first cluster
center1 = c(10,10)
#cov matrix
sigma = matrix(c(1,0.5,0.5,2),2,2)

llParser = function(log){
	info = log[grepl("newLL",log)]
	info = as.numeric(str_match(info,"newLL: (.+)")[,2]) #get log-likelihood values 
	return(info)
}

weightsParser = function(log){
	info = log[grepl("weights",log)]
	info = str_match(info,"weights: (.+)")[,2] #get weights values 
	info = lapply(info,function(x){eval(parse(text=x))})
	return(info)
}

meansParser = function(log){
	info = log[grepl("means",log)]
	info = str_match(info,"means: (.+)")[,2] #get means values 
	info = lapply(info,function(x){eval(parse(text=x))})
	return(info)
}

logParser = function(path){
	l = list()
	log = readLines(path)
	gmmInfo = log[grepl("modelPath",log)]

	l$ll = llParser(gmmInfo)
	l$weights = weightsParser(gmmInfo)
	l$means = meansParser(gmmInfo)

	return(l)
}

## plot means trajectory
saveMeansPlot = function(means,trueCenters){

	trueCentersDf = data.frame(Reduce(rbind,trueCenters))
	rownames(trueCentersDf) = NULL
	colnames(trueCentersDf) = c("x","y")

	k = length(l$means[[1]])
	meansPaths = list()
	for(i in 1:k){
		path = lapply(means,function(x){x[[i]]})
		path = data.frame(Reduce(rbind,path))
		colnames(path) = c("x","y")
		meansPaths[[i]] = path
	}

	meansPlot = ggplot(trueCentersDf,aes(x=x,y=y)) + geom_point(col="red",shape=18,size=6) + theme_minimal()

	for(i in 1:k){
		meansPlot = meansPlot + geom_point(data=meansPaths[[i]],aes(x=x,y=y),col="blue",size=1)
		meansPlot = meansPlot + geom_line(data=meansPaths[[i]],aes(x=x,y=y),col="blue",size=1)
	}

	meansPlot
	ggsave(filename="RUtils/pathPlots/means.pdf",meansPlot)
	message("means trajectories plot saved at RUtils/pathPlots")
}

## plot weights trajectory
displayWeightsPlot = function(weights,trueWeightsDf){
	weightsPathDf = data.frame((1+eps)*Reduce(rbind,weights))
	colnames(weightsPathDf) = c("x","y","z")


	zaxis <- list(
	  range = c(0,1)
	)

	x = 0:10/10
	y = x
	z = matrix(0,length(x),length(y))
	for(i in 1:nrow(z)){
		for(j in 1:ncol(z)){
			z[i,j] = 1 - x[i] - y[j]
		}
	}

	##simplex plot
	g = plot_ly(data = weightsPathDf, x = ~x, y = ~y, z = ~z, type="scatter3d", mode="lines+markers", marker = list(color="yellow",size=2), line = list(color="yellow")) %>% 
	add_surface(x=x,y=y,z=z,type="surface") %>%
	add_trace(data = trueWeightsDf,x= ~w2, y= ~w3, z= ~w1, type="scatter3d", marker = list(color="red",size=6)) %>%
	layout(xaxis = list(tickformat="d"), scene = list(zaxis=zaxis, aspectmode="cube")) 

	return(g)
}

saveLLPlot = function(ll){
	ll = ll[-1]
	df = data.frame(ll = ll, iter = 1:length(ll))
	g = ggplot(df,aes(x=iter,y=ll)) + 
	geom_point(col="darkblue") + 
	geom_line(col="darkblue") + 
	xlab("Iterations") + 
	ylab("Log-likelihood") + 
	theme_minimal()

	ggsave(filename="RUtils/pathPlots/ll.pdf",g)
	message("LL plot saved at RUtils/pathPlots")

}