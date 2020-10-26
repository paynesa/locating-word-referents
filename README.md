# Locating Word Referents
This code contains three modules corresponding to three models of word-referent matching in child language acquisition:
Propose-but-Verify, Pursuit, and Cross-Situational Learning. 

The `crosssituational` module provides an implementation of the modified Cross-Situational learner that uses 
`P(w|m)` in Stevens et al. (2017). This learner is a modified version of the one presented in Fazly et al. (2010).

The `proposebutverify` module provides an implementation of the Propose-but-Verify learner as proposed by 
Trueswell et al. (2013). Currently, only `a, a_0 = 1` are supported, since these are the parameters used by
Stevens et al. (2017).

Finally the `pursuit` module provides an implementation of the Pursuit learner proposed by Stevens et al. 2017. 
This learner additionally contains an optional sampling parameter allowing for the implementation of Pursuit with
Sampling or the original Pursuit learner. Pursuit with Sampling is like original Pursuit except that the meaning
retrieved for a word is retrieved based on the probabilities of each meaning, rather than deterministically selected; 
the original pursuit model always selects the meaning candidate with highest probability.

The `curricula` module provides the annotated Rollins data of child-directed speech along with objects in the scene
that is used by Stevens et al. (2017), as well as our models. 

## Running the Code

To run all learners with the parameters from Stevens et al. (2017), run: `python3 run_experiment.py`

To find the optimal parameters for Pursuit and run an experiment with these parameters, run: `python3 optimize_pursuit.py`.
This code currently optimizes original Pursuit but can easily be edited to optimize Pursuit with Sampling 
by changing both boolean values in the file to `True`. 

To find the optimal parameters for the Modified Cross-Situational learner, run `python3 optimize_xsit.py`
