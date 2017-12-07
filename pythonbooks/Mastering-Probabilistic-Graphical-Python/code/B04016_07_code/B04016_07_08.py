# creating a set of random observations
# As the observations can be one of the 5 states that is [0, 4],
# we can create them using np.random.randint
random_walk = np.random.randint(low=0, high=5, size=50)

# the array should be in the form of (n_observations, n_features)
# reshaping the array
random_walk = random_walk[:, np.newaxis]

# model.decode finds the most likely state sequence corresponding
# to the observation. By default it uses Viterbi algorithm
# it returns 2 parameters, the first one being log probability of
# the maximum likelihood path through the HMM and second being the
# state sequence.
logprob, state_sequence = model_multinomial.decode(random_walk)
