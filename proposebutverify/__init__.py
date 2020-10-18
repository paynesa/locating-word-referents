from typing import List, Tuple

class PbvLearner:
    """An implementation of Propose But Verify learning model (Trueswell et al. 2013).
    Takes in two parameters, a and a_0 as defined in Trueswell et al. 2013 and
    both defaulting to 1, following Stevens et al. 2017"""
    _alpha : int
    _alpha_naught : int
    def __init__(self, alpha : int = 1, alpha_naught : int = 1):
        self._alpha = alpha
        self._alpha_naught = alpha_naught

    def observe(self, curriculum: List[Tuple[str, List[str]]]):
        """Observe the given curriculum"""
        return