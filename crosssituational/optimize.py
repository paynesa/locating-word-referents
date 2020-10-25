from crosssituational import CrossSituationalLearner
from typing import Tuple
from curricula import load_train_test_curricula


def optimize_xsit() -> Tuple[int]:
    """Finds the best parameters for the modified cross-situational model"""
    # keep track of the current best parameters and corresponding results
    best_beta = 0
    best_lamdba = 0
    best_threshold = 0
    best_fscore = 0

    # load in the curricula
    print("Loading curricula...")
    train_curriculum, train_verification, test_curriculum, test_verification = (
        load_train_test_curricula()
    )

    for beta in [10, 100, 1000]:
        for lamda in [0.1, 0.01, 0.001, 0.0001]:
            # test every value of the threshold from 0.5-1 in increments of 0.01
            threshold: float = 0.0
            while threshold < 1.0:
                # get the average f score with these parameters over the instances
                print(f"Testing {beta} {lamda} {threshold}")
                learner = CrossSituationalLearner(
                    lambda_smoothing=lamda, beta=beta, tau_threshold=threshold
                )
                learner.observe(train_curriculum)
                precision, recall, f_score = learner.evaluate(train_verification)
                # check to see if we've found a new maximum f_score, update if we have
                if f_score > best_fscore:
                    best_fscore = f_score
                    best_beta = beta
                    best_lamdba = lamda
                    best_threshold = threshold
                threshold += 0.01

    return (best_beta, best_lamdba, best_threshold)
