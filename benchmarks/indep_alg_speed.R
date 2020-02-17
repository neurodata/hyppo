require("energy")
require("kernlab")
require("mgc")
require("HHG")
require("microbenchmark")

num_samples_range = c(50, 100, 200, 500, 1000, 2000, 5000, 10000)
linear_data <- list()
i <- 1
for (num_samples in num_samples_range){
  data <- mgc.sims.linear(num_samples, 1)
  x <- data$X
  y <- data$Y
  times = seq(1, 3, by=1)
  executions <- list()
  for (t in times){
    #x <- as.matrix(dist((x), diag = TRUE, upper = TRUE))
    #y <- as.matrix(dist((y), diag = TRUE, upper = TRUE))
    #time_taken <- microbenchmark(kmmd(x, y, ntimes=1000), times=1, unit="secs") # best of 5 executions
    #time_taken <- microbenchmark(dcor.test(x, y, R=1000), times=1, unit="secs") # best of 5 executions
    #time_taken <- microbenchmark(dcor.test(x, y, R=1000), times=1, unit="secs") # best of 5 executions
    time_taken <- microbenchmark(dcor2d(x, y), times=1, unit="secs") # best of 5 executions
    #time_taken <- microbenchmark(hhg.test(x, y, nr.perm = 1000), times=1, unit="secs") # best of 5 executions
    executions <- c(executions, list(time_taken[1, 2]/(10^9)))
  }
  linear_data <- c(linear_data, list(sapply(executions, mean)))
  
  print("Finished")
  i <- i + 1
}

df <- data.frame(matrix(unlist(linear_data), nrow=length(linear_data), byrow=T), stringsAsFactors=FALSE)
write.csv(df, row.names=FALSE)
