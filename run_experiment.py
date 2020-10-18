from curricula import get_curriculum, get_verification, load_train_test_curricula
from typing import Tuple
from proposebutverify import PbvLearner
from proposebutverify.run_pbv import run_propose_but_verify


print("Loading curricula...")
train_curriculum, train_verification, test_curriculum, test_verification = (
    load_train_test_curricula()
)
print("Running 1000 instances of Propose but Verify (Trueswell et al. 2017)...")
run_propose_but_verify(
    train_curriculum, train_verification, test_curriculum, test_verification, 1000
)
