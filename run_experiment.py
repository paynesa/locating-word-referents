from curricula import load_train_test_curricula
from pursuit import PursuitLearner
from pursuit.optimize import run_pursuit_experiment
from proposebutverify import PbvLearner
from crosssituational import CrossSituationalLearner


def run_experiment(
    learner_input,
    train_curriculum,
    train_verification,
    test_curriculum,
    test_verification,
    num_iterations: int = 1000,
    pursuit_sampling: bool = False,
):
    """Runs num_iterations of the propose-but-verify learner and prints out the
    precision, recall, and f-score for the training and testing data"""
    train_precision = 0
    train_recall = 0
    train_f_score = 0
    test_precision = 0
    test_recall = 0
    test_f_score = 0
    for i in range(num_iterations):
        # train and evaluate the learner on the training data
        if learner_input == PursuitLearner and pursuit_sampling:
            learner = learner_input(sample=True)
        else:
            learner = learner_input()
        learner.observe(train_curriculum)
        p, r, f = learner.evaluate(train_verification)
        train_precision += p
        train_recall += r
        train_f_score += f
        # train and evaluate the learner on the testing data
        if learner_input == PursuitLearner and pursuit_sampling:
            learner = learner_input(sample=True)
        else:
            learner = learner_input()
        learner.observe(test_curriculum)
        p, r, f = learner.evaluate(test_verification)
        test_precision += p
        test_recall += r
        test_f_score += f
    # average the precision, recall, and f-score for both training and testing
    train_precision = train_precision / num_iterations
    train_recall = train_recall / num_iterations
    train_f_score = train_f_score / num_iterations
    test_precision = test_precision / num_iterations
    test_recall = test_recall / num_iterations
    test_f_score = test_f_score / num_iterations
    # print out the averages for training and testing data
    print(
        f"\t Training average precision: {train_precision}, recall: {train_recall}, f-score: {train_f_score}"
    )
    print(
        f"\t Testing average precision: {test_precision}, recall: {test_recall}, f-score: {test_f_score}"
    )


print("Loading curricula...")
train_curriculum, train_verification, test_curriculum, test_verification = (
    load_train_test_curricula()
)

print("Running 1000 instances of Propose but Verify (Trueswell et al. 2017)...")
run_experiment(
    PbvLearner,
    train_curriculum,
    train_verification,
    test_curriculum,
    test_verification,
    1000,
)

print(
    "Running the Cross-Situational Learning model that makes use of P(w|m) (Stevens et al. 2017)..."
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
