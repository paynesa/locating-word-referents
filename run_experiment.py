from curricula import load_train_test_curricula
from proposebutverify.run_pbv import run_propose_but_verify
from pursuit import PursuitLearner

print("Loading curricula...")
train_curriculum, train_verification, test_curriculum, test_verification = (
    load_train_test_curricula()
)

learner = PursuitLearner()
learner.observe(train_curriculum)
learner.evaulate(train_verification)

# print("Running 1000 instances of Propose but Verify (Trueswell et al. 2017)...")
# run_propose_but_verify(
#     train_curriculum, train_verification, test_curriculum, test_verification, 1000
# )
