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

    def _update_maximum_strengths(self, objects : List[str]):
        updates = {}
        for word in self._hypotheses:
            for obj in self._hypotheses[word]:
                if obj in objects:
                    if obj not in updates or updates[obj] < self._hypotheses[word][obj]:
                        updates[obj] = self._hypotheses[word][obj]
        for update in updates:
            self._max_strengths[update] = updates[update]

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
        self._update_maximum_strengths([chosen_object])

    def _update_hypotheses(self, word: str, objects: List[str]):
        """Update the hypotheses based on an instance of learning"""
        sorted_meanings = [
            k
            for k, v in sorted(
                self._hypotheses[word].items(), key=lambda item: item[1], reverse=True
            )
        ]
        object_with_max_association = sorted_meanings[0]
        max_association_value = self._hypotheses[word][object_with_max_association]
        if object_with_max_association in objects:
            new_association = max_association_value + self._learning_rate * (
                1 - max_association_value
            )
            self._hypotheses[word][object_with_max_association] = new_association
        else:
            new_object: str = random.choice(objects)
            new_association = max_association_value * (1 - self._learning_rate)
            self._hypotheses[word][object_with_max_association] = new_association

            if new_object in self._hypotheses[word]:
                association_value_for_object = self._hypotheses[word][new_object]
                self._hypotheses[word][new_object] = (
                    association_value_for_object
                    + self._learning_rate * (1 - association_value_for_object)
                )
            else:
                self._hypotheses[word][new_object] = self._learning_rate
            self._update_maximum_strengths([new_object])
        self._update_maximum_strengths([object_with_max_association])

    def _get_conditional_probabilities(self, meanings: Dict[str, float])->Dict[str, float]:
        sum_Aw = sum(meanings.values())
        N = len(self._max_strengths)
        conditional_probabilities = {}
        for word in meanings:
            Awm = meanings[word]
            conditional_probabilities[word] = (Awm + self._smoothing_factor) / (sum_Aw + N*self._smoothing_factor)
        return conditional_probabilities

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        for (language, objects) in curriculum:
            for word in language.split():
                if word in self._hypotheses:
                    self._update_hypotheses(word, objects)
                else:
                    self._initialize(word, objects)
        hypotheses = {}
        for word in self._hypotheses:
            conditional_probabilities = self._get_conditional_probabilities(self._hypotheses[word])
            hypotheses_for_word = {k:v for k, v in conditional_probabilities.items() if v >= self._lexicalization_threshold}
            if hypotheses_for_word:
                hypotheses[word] = hypotheses_for_word
        self._hypotheses = hypotheses

    def evaulate(self, gold_standard: List[Tuple[str, str]]) -> Tuple[int]:
        """Get the precision, recall, and f-score when comparing to the gold standard"""
        true_positives: int = 0
        for (word, meaning) in gold_standard:
            if word in self._hypotheses and meaning in self._hypotheses[word]:
                true_positives += 1
        # precision = true positives / true positives + false positives
        precision: int = true_positives / len(self._hypotheses)
        # recall = true positives / true positives + false negatives
        recall: int = true_positives / len(gold_standard)
        f_score: int = 2 * (precision * recall) / (precision + recall)
        print(precision, recall, f_score)
        return (precision, recall, f_score)

