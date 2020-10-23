from pursuit import PursuitLearner
from curricula import load_train_test_curricula
from typing import Tuple


def optimize_pursuit(sample: bool, num_samples: int) -> Tuple[float]:
    """Finds the best paramaters for the pursuit learner over the number of samples"""

    # keep track of the current best parameters and corresponding results
    best_gamma = 0
    best_lamdba = 0
    best_threshold = 0
    best_fscore = 0

    # load in the curricula
    print("Loading curricula...")
    train_curriculum, train_verification, test_curriculum, test_verification = (
        load_train_test_curricula()
    )

    # possible gamma and lambda values as defined in Stevens et al. 2017
    for gamma in [0.01, 0.02, 0.05, 0.1]:
        for lamda in [0.1, 0.01, 0.001, 0.0001]:
            # test every value of the threshold from 0.5-1 in increments of 0.01
            threshold: float = 0.5
            while threshold < 1.0:
                # get the average f score with these parameters over the instances
                print(f"Testing {gamma} {lamda} {threshold}")
                f_score = 0
                for i in range(num_samples):
                    learner = PursuitLearner(
                        gamma_learning_rate=gamma,
                        lambda_smoothing=lamda,
                        tau_lexicalization=threshold,
                        sample=sample,
                    )
                    learner.observe(train_curriculum)
                    precision, recall, f = learner.evaluate(train_verification)
                    f_score += f
                f_score = f_score / num_samples
                # check to see if we've found a new maximum f_score, update if we have
                if f_score > best_fscore:
                    best_fscore = f_score
                    best_gamma = gamma
                    best_lamdba = lamda
                    best_threshold = threshold
                threshold += 0.01

    return (best_gamma, best_lamdba, best_threshold)


def run_pursuit_experiment(
    gamma_learning_rate: float,
    lambda_smothing: float,
    threshold: float,
    num_iterations: int = 1000,
    pursuit_sampling: bool = False,
):
    """Runs num_iterations of the pursuit learner and prints out the
    precision, recall, and f-score for the training and testing data"""
    print("Loading curricula...")
    train_curriculum, train_verification, test_curriculum, test_verification = (
        load_train_test_curricula()
    )
    train_precision = 0
    train_recall = 0
    train_f_score = 0
    test_precision = 0
    test_recall = 0
    test_f_score = 0
    for i in range(num_iterations):
        # train and evaluate the learner on the training data
        learner = PursuitLearner(
            gamma_learning_rate=gamma_learning_rate,
            lambda_smoothing=lambda_smothing,
            tau_lexicalization=threshold,
            sample=pursuit_sampling,
        )
        learner.observe(train_curriculum)
        p, r, f = learner.evaluate(train_verification)
        train_precision += p
        train_recall += r
        train_f_score += f
        # train and evaluate the learner on the testing data
        learner = PursuitLearner(
            gamma_learning_rate=gamma_learning_rate,
            lambda_smoothing=lambda_smothing,
            tau_lexicalization=threshold,
            sample=pursuit_sampling,
        )
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
