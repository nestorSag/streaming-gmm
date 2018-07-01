library(MASS)

#cov matrix
sigma = matrix(c(1,0.5,0.5,2),2,2)

#generate 500 points
datos1 = mvrnorm(n = 500, c(10,10), sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)

rotation = matrix(c(cos(2*pi/3),sin(2*pi/3),-sin(2*pi/3),cos(2*pi/3)),2,2)

#rotate points for the second cluster
datos2 = t(rotation %*% t(datos1[1:250,]))

#and again
datos3 = t(rotation %*% t(datos2[1:125,]))

#correct weights: 0.57,0.28,0.14
df = rbind(datos1,datos2,datos3)

write.table(df,"testdata.csv",sep=" ",row.names=FALSE,col.names=FALSE)