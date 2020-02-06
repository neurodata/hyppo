require("energy")
require("kernlab")
require("mgc")
require("microbenchmark")

num_samples_range = c(20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
linear_data <- list()
i <- 1
for (num_samples in num_samples_range){
  data <- mgc.sims.linear(num_samples, 1, eps=0.1)
  times = seq(1, 3, by=1)
  executions <- list()
  for (t in times){
    func <- kmmd
    #func <- dcov.test
    time_taken <- microbenchmark(func(data$X, data$Y), times=1, unit="secs") # best of 5 executions
    executions <- c(executions, list(time_taken[1, 2]/(10^9)))
  }
  linear_data <- c(linear_data, list(sapply(executions, mean)))
  
  print("Finished")
  i <- i + 1
}

print(linear_data)
