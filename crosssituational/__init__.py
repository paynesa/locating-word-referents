from typing import Tuple, List, Dict


class CrossSituationalLearner:
    """Cross-situational learner based on a modified version of Fazly et al. (2010)
    that considers P(w|m) rather than P(m|w), as proposed by Stevens et al. (2017)"""

    _smoothing: float
    _beta: float
    _threshold: float
    _hypotheses: Dict[str, Dict[str, float]]

    def __init__(
        self,
        lambda_smoothing: float = 0.01,
        beta: float = 100,
        tau_threshold: float = 0.09,
    ):
        """Initialize the model with the smmothing factor, beta, and threshold"""
        self._smoothing = lambda_smoothing
        self._beta = beta
        self._threshold = tau_threshold
        self._hypotheses = {}

    def _get_conditional_probability(self, word: str, meaning: str):
        """Get the conditional probability P(w|m) = [A(w, m) + lambda] /
        [sum for w’ in W (A(w’, m)) + beta*lambda]"""
        # get A(w, m) + lambda; this is the numerator
        numerator: float = self._smoothing + (
            self._hypotheses[word][meaning]
            if word in self._hypotheses and meaning in self._hypotheses[word]
            else 0
        )
        # intilalize the denominator to be beta*lambda, then iterate through and add each A(w', m)
        denominator = self._beta * self._smoothing
        for word in self._hypotheses:
            if meaning in self._hypotheses[word]:
                denominator += self._hypotheses[word][meaning]
        # return the conditional probability P(w, m)
        return numerator / denominator

    def _learn_from(self, word: str, objects: List[str]):
        """Learns from a given word and set of objects"""
        # add the word to our internal state of hypotheses
        if word not in self._hypotheses:
            self._hypotheses[word] = {}
        # get the conditional probabilities P(w|m) for each m in the scene
        probabilities: List[float] = [
            self._get_conditional_probability(word, object) for object in objects
        ]
        # for each object, increment its association by the alignment value
        i: int = 0
        while i < len(objects):
            if objects[i] not in self._hypotheses[word]:
                self._hypotheses[word][objects[i]] = 0
            # Increment association by Alignment(w, m) = P(w|m) / [sum for m’ in MU (P(w|m’))]
            self._hypotheses[word][objects[i]] += probabilities[i] / sum(probabilities)
            i += 1

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        for (language, objects) in curriculum:
            for word in language.split():
                self._learn_from(word, objects)
        final_hypotheses: Dict[str, Dict[str, float]] = {}
        for word in self._hypotheses:
            for meaning in self._hypotheses[word]:
                conditional_probability = self._get_conditional_probability(
                    word, meaning
                )
                if conditional_probability >= self._threshold:
                    if word not in final_hypotheses:
                        final_hypotheses[word] = {}
                    final_hypotheses[word][meaning] = conditional_probability
        self._hypotheses = final_hypotheses

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
