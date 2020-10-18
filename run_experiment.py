from curricula import get_curriculum
from proposebutverify import PbvLearner
curriculum = get_curriculum("curricula/rollins.txt")
learner = PbvLearner()
learner.observe(curriculum)