class PursuitLearner:
    """An implementation of the Pursuit learning model (Stevens et al. 2017).
    Takes in three parameters: gamma (the learning rate), lamda (the smoothing
    factor) and tau (the lexicalization threshold. These are set to defaults
    of 0.02, 0.001, and 0.79, respectively, following Stevens et al. (2017)"""

    _learning_rate : int
    _smoothing_factor : int
    _lexicalization_threshold : int
    def __init__(self, gamma_learning_rate : int = 0.02, lambda_smoothing : int = 0.001, tau_lexicalization : int = 0.79):
        self._learning_rate = gamma_learning_rate
        self._smoothing_factor = lambda_smoothing
        self._lexicalization_threshold = tau_lexicalization
        