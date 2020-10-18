from curricula import get_curriculum, get_verification
from proposebutverify import PbvLearner
print("Loading curricula...")
train_curriculum = get_curriculum("curricula/train.txt")
train_verification = get_verification("curricula/train.gold")
test_curriculum = get_curriculum("curricula/test.txt")
test_verification = get_verification("curricula/test.gold")
print("Running 1000 instances of Propose but Verify (Trueswell et al. 2017)...")
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
print(f"\t Training average precision: {train_precision}, recall: {train_recall}, f-score: {train_f_score}")
print(f"\t Testing average precision: {test_precision}, recall: {test_recall}, f-score: {test_f_score}")






