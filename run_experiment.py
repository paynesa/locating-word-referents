from curricula import get_curriculum, get_verification
from proposebutverify import PbvLearner
train_curriculum = get_curriculum("curricula/train.txt")
train_verification = get_verification("curricula/train.gold")
test_curriculum = get_curriculum("curricula/test.txt")
test_verification = get_curriculum("curricula/test.gold")
learner = PbvLearner()
precision = 0
recall = 0
f_score = 0
for i in range(1000):
    learner.observe(train_curriculum)
    p, r, f = learner.evaluate(train_verification)
    precision += p
    recall += r
    f_score += f
precision = precision / 1000
recall = recall / 1000
f_score = f_score / 1000
print(precision, recall, f_score)





