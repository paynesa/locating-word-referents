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
        tau_threshold: float = 0.57,
    ):
        """Initialize the model with the smmothing factor, beta, and threshold"""
        self._smoothing = lambda_smoothing
        self._beta = beta
        self._threshold = tau_threshold
        self._hypotheses = {}

    def _get_conditional_probability(self, word: str, meaning: str):
        numerator: float = self._smoothing + (
            self._hypotheses[word][meaning]
            if word in self._hypotheses and meaning in self._hypotheses[word]
            else 0
        )
        denominator = self._beta * self._smoothing
        for word in self._hypotheses:
            if meaning in self._hypotheses[word]:
                denominator += self._hypotheses[word][meaning]
        return numerator/denominator

    def _learn_from(self, word: str, objects: List[str]):
        if word not in self._hypotheses:
            self._hypotheses[word] = {}
        alignments : List[float] = [self._get_conditional_probability(word, object) for object in objects]
        i : int = 0
        while i < len(objects):
            if objects[i] not in self._hypotheses[word]:
                self._hypotheses[word][objects[i]] = 0
            # Alignment(w, m) = P(w|m) / [sum for m’ in MU (P(w|m’))]
            self._hypotheses[word][objects[i]] += alignments[i]/sum(alignments)
            i += 1

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe and learn from the given curriculum"""
        for (language, objects) in curriculum:
            for word in language.split():
                self._learn_from(word, objects)
        for word in self._hypotheses:
            print(word, self._hypotheses[word])
