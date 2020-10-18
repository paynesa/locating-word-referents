from curricula import get_curriculum
from proposebutverify import PbvLearner
curriculum = get_curriculum("curricula/train.txt")
learner = PbvLearner()
learner.observe(curriculum)
print(len(curriculum))
print(learner._hypotheses)