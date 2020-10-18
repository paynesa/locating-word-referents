from typing import List, Tuple, Dict
import random

#TODO: handle non-1 alpha values

class PbvLearner:
    """An implementation of Propose But Verify learning model (Trueswell et al. 2013).
    Takes in two parameters, a and a_0 as defined in Trueswell et al. 2013 and
    both defaulting to 1, following Stevens et al. 2017"""
    _alpha : int
    _alpha_naught : int
    _hypotheses : Dict[str, str]
    def __init__(self, alpha : int = 1, alpha_naught : int = 1):
        self._alpha = alpha
        self._alpha_naught = alpha_naught
        self._hypotheses = {}

    def _verify_meaning(self, word: str, objects : List[str]):
        """Verify the meaning of a word already in the lexicon. If this meaning
        isn't verified, pick a new meaning from the situation to append to the
        lexicon instead"""
        object_to_verify : str = self._hypotheses[word]
        if object_to_verify not in objects:
            self._select_meaning(word, objects)

    def _select_meaning(self, word: str, objects : List[str])->None:
        """Select a new meaning at random for a word that hasn't been observed
        before"""
        self._hypotheses[word] = random.choice(objects)

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe the given curriculum"""
        for (language, objects) in curriculum:
            for word in language.split():
                if word in self._hypotheses:
                    self._verify_meaning(word, objects)
                else:
                    self._select_meaning(word, objects)
        return