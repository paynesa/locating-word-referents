from crosssituational.optimize import optimize_xsit

print("Getting optimized paramters over 100 instances for pursuit without sampling...")
best_beta, best_lamda, best_threshold = optimize_xsit()
print(f"\nBest parameters: {best_gamma}, {best_lamda}, {best_threshold}")
