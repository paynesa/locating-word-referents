from curricula import get_curriculum, get_verification
from proposebutverify import PbvLearner
train_curriculum = get_curriculum("curricula/train.txt")
train_verification = get_verification("curricula/train.gold")
test_curriculum = get_curriculum("curricula/test.txt")
test_verification = get_verification("curricula/test.gold")
learner = PbvLearner()
train_precision = 0
train_recall = 0
train_f_score = 0
test_precision = 0
test_recall = 0
test_f_score = 0
for i in range(1000):
    learner = PbvLearner()
    learner.observe(train_curriculum)
    p, r, f = learner.evaluate(train_verification)
    train_precision += p
    train_recall += r
    train_f_score += f
    learner = PbvLearner()
    learner.observe(test_curriculum)
    p, r, f = learner.evaluate(test_verification)
    test_precision += p
    test_recall += r
    test_f_score += f
train_precision = train_precision / 1000
train_recall = train_recall / 1000
train_f_score = train_f_score / 1000
test_precision = test_precision / 1000
test_recall = test_recall / 1000
test_f_score = test_f_score / 1000
print(train_precision, train_recall, train_f_score)
print(test_precision, test_recall, test_f_score)





