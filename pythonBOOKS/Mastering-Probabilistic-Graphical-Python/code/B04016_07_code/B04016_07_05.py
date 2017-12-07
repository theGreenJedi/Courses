from hmmlearn.hmm import MultinomialHMM
import numpy as np

# Here n_components correspond to number of states in the hidden
# variables and n_symbols correspond to number of states in the
# obversed variables
model_multinomial = MultinomialHMM(n_components=4)

# Transition probability as specified above
transition_matrix = np.array([[0.2, 0.6, 0.15, 0.05],
                              [0.2, 0.3, 0.3, 0.2],
                              [0.05, 0.05, 0.7, 0.2],
                              [0.005, 0.045, 0.15, 0.8]])

# Setting the transition probability
model_multinomial.transmat_ = transition_matrix

# Initial state probability
initial_state_prob = np.array([0.1, 0.4, 0.4, 0.1])

# Setting initial state probability
model_multinomial.startprob_ = initial_state_prob

# Here the emission prob is required to be in the shape of
# (n_components, n_symbols). So instead of directly feeding the
# CPD we would using the transpose of it.
emission_prob = np.array([[0.045, 0.15, 0.2, 0.6, 0.005],
                          [0.2, 0.2, 0.2, 0.3, 0.1],
                          [0.3, 0.1, 0.1, 0.05, 0.45],
                          [0.1, 0.1, 0.2, 0.05, 0.55]])

# Setting the emission probability
model_multinomial.emissionprob_ = emission_prob

# model.sample returns both observations as well as hidden states
# the first return argument being the observation and the second
# being the hidden states
Z, X = model_multinomial.sample(100)
