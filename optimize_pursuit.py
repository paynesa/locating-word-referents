from pursuit.optimize import optimize_pursuit

best_gamma, best_lamda, best_threshold = optimize_pursuit(True, 1)
print(best_gamma, best_lamda, best_threshold)
