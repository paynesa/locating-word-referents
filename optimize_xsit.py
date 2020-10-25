from crosssituational.optimize import optimize_xsit
from crosssituational import CrossSituationalLearner
from curricula import load_train_test_curricula

# get the optimized paramters
print("Getting optimized paramters over 100 instances for pursuit without sampling...")
best_beta, best_lamda, best_threshold = optimize_xsit()
print(f"\nBest parameters: {best_beta}, {best_lamda}, {best_threshold}")

print(
    f"Running cross-situational experiment with beta={best_beta}, lamda={best_lamda}, threshold={best_threshold}"
)

# load in the training and testing data
train_curriculum, train_verification, test_curriculum, test_verification = (
    load_train_test_curricula()
)

# test on the training curricula first
learner = CrossSituationalLearner(
    beta=best_beta, lambda_smoothing=best_lamda, tau_threshold=best_threshold
)
learner.observe(train_curriculum)
train_precision, train_recall, train_f = learner.evaluate(train_verification)
print(
    f"\t Training average precision: {train_precision}, recall: {train_recall}, f-score: {train_f}"
)

# now on the testing curricula
learner = CrossSituationalLearner(
    beta=best_beta, lambda_smoothing=best_lamda, tau_threshold=best_threshold
)
learner.observe(test_curriculum)
test_precision, test_recall, test_f = learner.evaluate(test_verification)
print(
    f"\t Testing average precision: {test_precision}, recall: {test_recall}, f-score: {test_f}"
)
