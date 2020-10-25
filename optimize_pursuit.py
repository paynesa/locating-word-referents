from pursuit.optimize import optimize_pursuit, run_pursuit_experiment

print("Getting optimized parameters over 100 instances for pursuit without sampling...")
best_gamma, best_lamda, best_threshold = optimize_pursuit(False, 100)
print(f"\nBest parameters: {best_gamma}, {best_lamda}, {best_threshold}")

print(
    f"Running pursuit experiments with gamma={best_gamma}, lamda={best_lamda}, threshold={best_threshold}"
)
run_pursuit_experiment(
    gamma_learning_rate=best_gamma,
    lambda_smothing=best_lamda,
    threshold=best_threshold,
    num_iterations=1000,
    pursuit_sampling=False,
)
