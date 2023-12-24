import random
from curricula import load_rollins
from crosssituational import CrossSituationalLearner
from proposebutverify import PbvLearner
from pursuit import PursuitLearner
import numpy as np


def run_xsit(train, test):
    """Run the Cross-Situational Learner"""
    print("Running the Cross-Situational Learning model...")
    learner = CrossSituationalLearner()
    learner.observe(train)
    precision, recall, f = learner.evaluate(test)
    print(
        f"\tprecision: {precision :.3f}, recall: {recall :.3f}, f-score: {f :.3f}"
    )


def run_pbv(train, test, iters=1000):
    """Run the PbV learner"""
    print("Running the Propose but Verify Learning model...")
    precisions = []
    recalls = []
    fs = []
    for i in range(iters):
        learner = PbvLearner()
        learner.observe(train)
        p, r, f = learner.evaluate(test)
        precisions.append(p)
        recalls.append(r)
        fs.append(f)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    fs = np.asarray(fs)
    p_mean = np.mean(precisions)
    p_std = np.std(precisions)
    r_mean = np.mean(recalls)
    r_std = np.std(recalls)
    f_mean = np.mean(fs)
    f_std = np.std(fs)
    print(
        f"\tprecision: {p_mean :.3f} ({p_std :.3f}), recall: {r_mean :.3f} ({r_std :.3f}), f-score: {f_mean :.3f} ({f_std :.3f})"
    )


def run_pursuit(train, test, iters=1000, sampling=True):
    """Run the Pursuit Learner"""
    if sampling:
        print("Running the Pursuit Learning Model with Sampling...")
    else:
        print("Running the Pursuit Learning Model without Sampling...")
    precisions = []
    recalls = []
    fs = []
    for i in range(iters):
        learner = PursuitLearner(sample=sampling)
        learner.observe(train)
        p, r, f = learner.evaluate(test)
        precisions.append(p)
        recalls.append(r)
        fs.append(f)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    fs = np.asarray(fs)
    p_mean = np.mean(precisions)
    p_std = np.std(precisions)
    r_mean = np.mean(recalls)
    r_std = np.std(recalls)
    f_mean = np.mean(fs)
    f_std = np.std(fs)
    print(
        f"\tprecision: {p_mean :.3f} ({p_std :.3f}), recall: {r_mean :.3f} ({r_std :.3f}), f-score: {f_mean :.3f} ({f_std :.3f})"
    )


if __name__ == "__main__":
    train, test = load_rollins()
    run_pbv(train, test)
    run_xsit(train, test)
    run_pursuit(train, test, sampling=False)
    run_pursuit(train, test, sampling=True)
    print("SHUFFLING TRAIN...")
    random.shuffle(train)
    run_pbv(train, test)
    run_xsit(train, test)
    run_pursuit(train, test, sampling=False)
    run_pursuit(train, test, sampling=True)
