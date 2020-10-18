from typing import List, Tuple, Dict
import random


class PursuitLearner:
    """An implementation of the Pursuit learning model (Stevens et al. 2017).
    Takes in three parameters: gamma (the learning rate), lamda (the smoothing
    factor) and tau (the lexicalization threshold. These are set to defaults
    of 0.02, 0.001, and 0.79, respectively, following Stevens et al. (2017)"""

    _learning_rate: int
    _smoothing_factor: int
    _lexicalization_threshold: int
    _hypotheses: Dict[str, Dict[str, int]]
    _max_strengths: Dict[str, int]

    def __init__(
        self,
        gamma_learning_rate: int = 0.02,
        lambda_smoothing: int = 0.001,
        tau_lexicalization: int = 0.79,
    ):
        """Initialize a pursuit learner with the given learning rate, smoothing factor, and lexicalization threshold"""
        self._learning_rate = gamma_learning_rate
        self._smoothing_factor = lambda_smoothing
        self._lexicalization_threshold = tau_lexicalization
        self._hypotheses = {}
        self._max_strengths = {}

    def _initialize(self, word: str, objects: List[str]):
        """Initialize the hypothesis for a word that has never been seen before,
        using mutual exclusion to pick the one with the smallest existing association"""
        # get the maxiumum assocation strengths for each object
        association_strengths: Dict = {
            object: self._max_strengths[object] if object in self._max_strengths else 0
            for object in objects
        }
        # get the minimum association strength of all of them to maintain mutual exclusivity
        min_strength = [
            v
            for k, v in sorted(association_strengths.items(), key=lambda item: item[1])
        ][0]
        # if there are multiple objects with this association strength, choose one at random
        chosen_object = random.choice(
            [
                object
                for object in association_strengths
                if association_strengths[object] == min_strength
            ]
        )
        # update the hypotheses for the word
        self._hypotheses[word] = {}
        self._hypotheses[word][chosen_object] = self._learning_rate
        # if the object doesn't already have a maximum association strength, update it
        if chosen_object not in self._max_strengths:
            self._max_strengths[chosen_object] = self._learning_rate

    def _update_hypotheses(self, word : str, objects : List[str]):
        """Update the hypotheses based on an instance of learning"""
        sorted_meanings = [k for k, v in sorted(self._hypotheses[word].items(), key=lambda item: item[1], reverse=True)]
        object_with_max_association = sorted_meanings[0]
        max_association_value = self._hypotheses[word][object_with_max_association]
        if object_with_max_association in objects:
            new_association = max_association_value + self._learning_rate*(1 - max_association_value)
            self._hypotheses[word][object_with_max_association] = new_association
            if object_with_max_association not in self._max_strengths or self._max_strengths[object_with_max_association] < new_association:
                self._max_strengths[object_with_max_association] = new_association
        else :
            new_object : str = random.choice(objects)
            new_association = max_association_value*(1-self._learning_rate)
            self._hypotheses[word][
                object_with_max_association] = new_association
            if object_with_max_association in self._max_strengths and self._max_strengths[object_with_max_association] == max_association_value:
                self._max_strengths[object_with_max_association] = new_association
            if new_object in self._hypotheses[word]:
                association_value_for_object = self._hypotheses[word][new_object]
                self._hypotheses[word][new_object] = association_value_for_object + self._learning_rate*(1-association_value_for_object)
                if new_object not in self._max_strengths or self._max_strengths[new_object] < self._hypotheses[word][new_object]:
                    self._max_strengths[new_object] = self._hypotheses[word][new_object]
            else:
                self._hypotheses[word][new_object] = self._learning_rate
                if new_object not in self._max_strengths or self._max_strengths[new_object] < self._learning_rate:
                    self._max_strengths[new_object] = self._learning_rate





    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        for (language, objects) in curriculum:
            for word in language.split():
                if word in self._hypotheses:
                    self._update_hypotheses(word, objects)
                else:
                    self._initialize(word, objects)

        return

    def evaulate(self, gold_standard: List[Tuple[str, str]]) -> Tuple[int]:
        """Get the precision, recall, and f-score when comparing to the gold standard"""
        return
