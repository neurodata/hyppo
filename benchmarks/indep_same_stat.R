require("energy")
require("kernelPSI")
require("kpcalg")
require("mgc")
require("HHG")
# change to your file path
filepath = setwd("/Users/sampan501/workspace/mgc/benchmarks/same_stat/spiral")

times = seq(1, 20, by=1)
statistics <- list()
for (t in times){
  df <- read.csv(paste(filepath, "/", t, ".csv", sep=""), sep=",")
  x <- df[, 1]
  y <- df[, 2]
  #stat <- mgc.test(x, y)
  stat <- bcdcor(x, y)
  #Dx = as.matrix(dist((x), diag = TRUE, upper = TRUE))
  #Dy = as.matrix(dist((y), diag = TRUE, upper = TRUE))
  #stat <- hhg.test(Dx, Dy)$sum.chisq
  statistics <- c(statistics, list(stat))
}

df <- data.frame(matrix(unlist(statistics), nrow=length(statistics), byrow=T), stringsAsFactors=FALSE)
write.csv(df, row.names=FALSE)
