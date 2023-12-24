from curricula import load_train_test_curricula
from pursuit.optimize import run_pursuit_experiment
from proposebutverify.run_pbv import run_pbv
from crosssituational import CrossSituationalLearner

print("Running 1000 instances of Propose but Verify (Trueswell et al. 2017)...")
run_pbv(1000)

print(
    "Running the Cross-Situational Learning model that makes use of P(w|m) (Stevens et al. 2017)..."
)
train_curriculum, train_verification, test_curriculum, test_verification = (
    load_train_test_curricula()
)
learner = CrossSituationalLearner()
learner.observe(train_curriculum)
train_precision, train_recall, train_f = learner.evaluate(train_verification)
print(
    f"\t Training average precision: {train_precision}, recall: {train_recall}, f-score: {train_f}"
)
learner = CrossSituationalLearner()
learner.observe(test_curriculum)
test_precision, test_recall, test_f = learner.evaluate(test_verification)
print(
    f"\t Testing average precision: {test_precision}, recall: {test_recall}, f-score: {test_f}"
)


print("Running 1000 instances of Pursuit (Stevens et al. 2017)...")
run_pursuit_experiment(
    gamma_learning_rate=0.02,
    lambda_smothing=0.001,
    threshold=0.79,
    num_iterations=1000,
    pursuit_sampling=False,
)


print("Running 1000 instances of Pursuit with sampling...")
run_pursuit_experiment(
    gamma_learning_rate=0.02,
    lambda_smothing=0.001,
    threshold=0.79,
    num_iterations=1000,
    pursuit_sampling=True,
)
