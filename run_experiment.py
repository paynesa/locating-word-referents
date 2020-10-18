from curricula import get_curriculum, get_verification
from proposebutverify import PbvLearner
curriculum = get_curriculum("curricula/train.txt")
verification = get_verification("curricula/train.gold")
learner = PbvLearner()
learner.observe(curriculum)
precision, recall, f_score = learner.evaluate(verification)
print(precision, recall, f_score)
