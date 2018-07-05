source("RUtils/utils.R")
library(MASS)


#generate 500 points
datos1 = mvrnorm(n = 500, center1, sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)

#rotate points for the second cluster
datos2 = t(rotation %*% t(datos1[1:250,]))

#and again
datos3 = t(rotation %*% t(datos2[1:125,]))

#correct weights: ~ 0.57,0.28,0.14
df = rbind(datos1,datos2,datos3)

write.table(df,testDataPath,sep=" ",row.names=FALSE,col.names=FALSE)