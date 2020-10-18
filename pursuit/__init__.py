from typing import List, Tuple, Dict
import random


class PursuitLearner:
    """An implementation of the Pursuit learning model (Stevens et al. 2017).
    Takes in three parameters: gamma (the learning rate), lamda (the smoothing
    factor) and tau (the lexicalization threshold. These are set to defaults
    of 0.02, 0.001, and 0.79, respectively, following Stevens et al. (2017)"""

    _learning_rate: float
    _smoothing_factor: float
    _lexicalization_threshold: float
    _hypotheses: Dict[str, Dict[str, float]]
    _max_strengths: Dict[str, float]

    def __init__(
        self,
        gamma_learning_rate: float = 0.02,
        lambda_smoothing: float = 0.001,
        tau_lexicalization: float = 0.79,
    ):
        """Initialize a pursuit learner with the given learning rate, smoothing factor, and lexicalization threshold"""
        self._learning_rate = gamma_learning_rate
        self._smoothing_factor = lambda_smoothing
        self._lexicalization_threshold = tau_lexicalization
        self._hypotheses = {}
        self._max_strengths = {}

    def _update_maximum_strengths(self, object: str):
        """This updates a dictionary mapping meanings to the maximum strengths for each meaning.
        We update it each time a weight is updated for a given meaning in order to keep the
        most up-to-date maximum weights for mutual exclusion during initialization"""
        updates: Dict[str, float] = {}
        # iterate through the hypothesized meanings for each word
        for word in self._hypotheses:
            for obj in self._hypotheses[word]:
                if obj == object:
                    # if we have a new maximum assocation for the meaning, update the dictionary
                    if obj not in updates or updates[obj] < self._hypotheses[word][obj]:
                        updates[obj] = self._hypotheses[word][obj]
        # update the max strengths dictionary to reflect these updates
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
        # update the maximum association strength of the object
        self._update_maximum_strengths(chosen_object)

    def _update_hypotheses(self, word: str, objects: List[str]):
        """Update the hypotheses based on an instance of learning"""
        # get the object with maximum association for a given word
        sorted_meanings = [
            k
            for k, v in sorted(
                self._hypotheses[word].items(), key=lambda item: item[1], reverse=True
            )
        ]
        object_with_max_association = sorted_meanings[0]
        # get the association value for this object
        max_association_value = self._hypotheses[word][object_with_max_association]
        # reward the object if it is in the observed objects
        if object_with_max_association in objects:
            new_association = max_association_value + self._learning_rate * (
                1 - max_association_value
            )
            self._hypotheses[word][object_with_max_association] = new_association
        # otherwise, penalize it
        else:
            new_association = max_association_value * (1 - self._learning_rate)
            self._hypotheses[word][object_with_max_association] = new_association
            # select a new object at random to be the chosen object in the scene
            new_object: str = random.choice(objects)
            # if we have already hypothesized this object for this word, reward
            # the existing association
            if new_object in self._hypotheses[word]:
                association_value_for_object = self._hypotheses[word][new_object]
                self._hypotheses[word][new_object] = (
                    association_value_for_object
                    + self._learning_rate * (1 - association_value_for_object)
                )
            # otherwise, set the association equal to the learning rate
            else:
                self._hypotheses[word][new_object] = self._learning_rate
            # update the maximum strengths corresponding to the new object
            self._update_maximum_strengths(new_object)
        # update the maximum strengths corresponding to the object that was either rewarded or penalized
        self._update_maximum_strengths(object_with_max_association)

    def _get_conditional_probabilities(
        self, meanings: Dict[str, float]
    ) -> Dict[str, float]:
        """Get the conditional probabilities P(m|w) for the dictionary of meaning : weight mappings"""
        sum_Aw = sum(meanings.values())
        N = len(self._max_strengths)
        conditional_probabilities: Dict[str, float] = {}
        for word in meanings:
            Awm = meanings[word]
            # calculation of conditional probabilities from Stevens et al. 2017
            conditional_probabilities[word] = (Awm + self._smoothing_factor) / (
                sum_Aw + N * self._smoothing_factor
            )
        return conditional_probabilities

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        # learn from each instance in the curriculum
        for (language, objects) in curriculum:
            for word in language.split():
                if word in self._hypotheses:
                    self._update_hypotheses(word, objects)
                else:
                    self._initialize(word, objects)
        # now, update the hypotheses to be those for which P(m|w) > Tau
        hypotheses: Dict[str, Dict[str, float]] = {}
        for word in self._hypotheses:
            # get the conditional probabilities, P(m|w)
            conditional_probabilities = self._get_conditional_probabilities(
                self._hypotheses[word]
            )
            # get those for which P(m|w) > threshold, and update the hypotheses accordingly
            hypotheses_for_word = {
                k: v
                for k, v in conditional_probabilities.items()
                if v >= self._lexicalization_threshold
            }
            if hypotheses_for_word:
                hypotheses[word] = hypotheses_for_word
        # set the global hypotheses equal to the filtered ones
        self._hypotheses = hypotheses

    def evaluate(self, gold_standard: List[Tuple[str, str]]) -> Tuple[float]:
        """Get the precision, recall, and f-score when comparing to the gold standard"""
        true_positives: int = 0
        for (word, meaning) in gold_standard:
            if word in self._hypotheses and meaning in self._hypotheses[word]:
                true_positives += 1
        # precision = true positives / true positives + false positives
        precision: float = true_positives / len(self._hypotheses)
        # recall = true positives / true positives + false negatives
        recall: float = true_positives / len(gold_standard)
        f_score: float = 2 * (precision * recall) / (precision + recall)
        return (precision, recall, f_score)
