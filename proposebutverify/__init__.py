from typing import List, Tuple, Dict
import random


class PbvLearner:
    """An implementation of Propose But Verify learning model (Trueswell et al. 2013).
    Takes in two parameters, a and a_0 as defined in Trueswell et al. 2013 and
    both defaulting to 1, following Stevens et al. 2017"""

    _alpha: float
    _alpha_naught: float
    _hypotheses: Dict[str, str]

    def __init__(self, alpha: float = 1, alpha_naught: float = 1):
        self._alpha = alpha
        self._alpha_naught = alpha_naught
        self._hypotheses = {}
        self._verified = set()

    def _verify_meaning(self, word: str, objects: List[str]):
        """Verify the meaning of a word already in the lexicon. If this meaning
        isn't verified, pick a new meaning from the situation to append to the
        lexicon instead"""
        object_to_verify: str = self._hypotheses[word]
        # if the word hasn't been verified yet, then the probability of retrieval is alpha_0
        if word not in self._verified:
            remember_alpha = random.choices([0, 1], weights=[1 - self._alpha_naught, self._alpha_naught], k=1)[0]
            # if we don't remember it or we haven't seen it, then select a new one at random
            if object_to_verify not in objects or remember_alpha == 0:
                self._select_meaning(word, objects)
            # otherwise, we've verified it for the first time
            else:
                self._verified.add(word)
        else:
            # if we don't remember it or we haven't seen it, then select a new one at random, and it's not verified
            remember_alpha = random.choices([0, 1], weights=[1-self._alpha, self._alpha], k=1)[0]
            if object_to_verify not in objects or remember_alpha == 0:
                self._select_meaning(word, objects)
                self._verified.remove(word)

    def _select_meaning(self, word: str, objects: List[str]) -> None:
        """Select a new meaning at random for a word that hasn't been observed
        before"""
        self._hypotheses[word] = random.choice(objects)

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        for (language, objects) in curriculum:
            # try to learn a meaning for each word
            for word in language.split():
                # if there is already a hypothesis, verify it
                if word in self._hypotheses:
                    # keep track of if this is the first time we're verifying or not
                    self._verify_meaning(word, objects)
                # otherwise, select an object at random
                else:
                    self._select_meaning(word, objects)

    def evaluate(self, gold_standard: List[Tuple[str, str]]) -> Tuple[float]:
        """Get the precision, recall, and f-score when comparing to the gold standard"""
        correct: int = 0
        for (word, meaning) in gold_standard:
            if word in self._hypotheses and self._hypotheses[word] == meaning:
                correct += 1
        # precision = correct / items in lexicon
        precision: float = correct / len(self._hypotheses)
        # recall = correct / items in the gold
        recall: float = correct / len(gold_standard)
        f_score: float = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f_score
