rm(list=ls())

require("energy")
require("kernlab")
require("HHG")
# change to your file path
#filepath = setwd("/Users/sampan501/workspace/hyppo/benchmarks/same_stat/indep")
filepath = setwd("/Users/sampan501/workspace/hyppo/benchmarks/same_stat/ksample")

times = seq(1, 100, by=1)
statistics <- list()
for (t in times){
  #df <- read.csv(paste(filepath, "/", t, ".csv", sep=""), sep=",")
  df1 <- read.csv(paste(filepath, "/sim1_", t, ".csv", sep=""), sep=",")
  df2 <- read.csv(paste(filepath, "/sim2_", t, ".csv", sep=""), sep=",")
  #x <- df[, 1]
  #y <- df[, 2]
  x <- df1[,]
  y <- df2[,]
  #stat <- bcdcor(x, y)
  #Dx = as.matrix(dist((x), diag = TRUE, upper = TRUE))
  #Dy = as.matrix(dist((y), diag = TRUE, upper = TRUE))
  #stat <- hhg.test(Dx, Dy, nr.perm=0)$sum.chisq
  stat <- kmmd(x, y, ntimes=0)@mmdstats[2]
  statistics <- c(statistics, list(stat))
}

df <- data.frame(matrix(unlist(statistics), nrow=length(statistics), byrow=T), stringsAsFactors=FALSE)
write.csv(df, row.names=FALSE)
