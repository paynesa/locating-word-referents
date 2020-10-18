# TODO: currently only supports alpha=1

from typing import List, Tuple, Dict
import random


class PbvLearner:
    """An implementation of Propose But Verify learning model (Trueswell et al. 2013).
    Takes in two parameters, a and a_0 as defined in Trueswell et al. 2013 and
    both defaulting to 1, following Stevens et al. 2017"""

    _alpha: int
    _alpha_naught: int
    _hypotheses: Dict[str, str]

    def __init__(self, alpha: int = 1, alpha_naught: int = 1):
        self._alpha = alpha
        self._alpha_naught = alpha_naught
        self._hypotheses = {}

    def _verify_meaning(self, word: str, objects: List[str]):
        """Verify the meaning of a word already in the lexicon. If this meaning
        isn't verified, pick a new meaning from the situation to append to the
        lexicon instead"""
        object_to_verify: str = self._hypotheses[word]
        if object_to_verify not in objects:
            self._select_meaning(word, objects)

    def _select_meaning(self, word: str, objects: List[str]) -> None:
        """Select a new meaning at random for a word that hasn't been observed
        before"""
        self._hypotheses[word] = random.choice(objects)

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe the given curriculum"""
        for (language, objects) in curriculum:
            # try to learn a meaning for each word
            for word in language.split():
                # if there is already a hypothesis, verify it
                if word in self._hypotheses:
                    self._verify_meaning(word, objects)
                # otherwise, select an object at random
                else:
                    self._select_meaning(word, objects)

    def evaluate(self, gold_standard: List[Tuple[str, str]]) -> Tuple[int]:
        """Get the precision, recall, and f-score when comparing to the gold standard"""
        true_positives: int = 0
        for (word, meaning) in gold_standard:
            if word in self._hypotheses and self._hypotheses[word] == meaning:
                true_positives += 1
        # precision = true positives / true positives + false positives
        precision: int = true_positives / len(self._hypotheses)
        # recall = true positives / true positives + false negatives
        recall: int = true_positives / len(gold_standard)
        f_score: int = 2 * (precision * recall) / (precision + recall)
        return (precision, recall, f_score)
