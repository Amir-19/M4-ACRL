"""
solving pendulum using actor-critic model
"""

import numpy as np

class ActorCriticContinuous:

    def __init__(self, num_state, stepsize_mu = 0.1, stepsize_sigma = 0.01, stepsize_v = 0.1, stepsize_r = 0.1):

        # step sizes
        self.stepsize_v = stepsize_v
        self.stepsize_mu = stepsize_mu
        self.stepsize_sigma = stepsize_sigma
        self.stepsize_r = stepsize_r

        # num of features
        self.num_state = num_state

        # average reward
        self.R_bar = 0.0

        # elegibility traces
        self.e_v = np.zeros(self.num_state)
        self.e_mu = np.zeros(self.num_state)
        self.e_sigma = np.zeros(self.num_state)

        # weights
        self.w_v = np.zeros(self.num_state)
        self.w_mu = np.zeros(self.num_state)
        self.w_sigma = np.zeros(self.num_state)

        # latest action
        self.action = None

    def update(self, state, reward, state_prime, lam):

        delta = reward - self.R_bar + np.dot(self.w_v, state_prime) - np.dot(self.w_v, state)
        self.update_critic(state, lam, delta)
        mean_actor, sigma_actor = self.update_actor(state, lam, delta)
        return mean_actor, sigma_actor

    def update_critic(self, state, lam, delta):

        self.R_bar += self.stepsize_r * delta
        self.e_v = self.e_v * lam + state
        self.w_v += self.stepsize_v * delta * self.e_v

    def update_actor(self, state, lam, delta):

        mean = np.dot(self.w_mu, state)
        sigma = np.exp(np.dot(self.w_sigma, state) + 0.000001)

        self.e_mu = self.e_mu * lam + (self.action - mean) * state
        self.w_mu += self.stepsize_mu * self.e_mu * delta

        self.e_sigma = self.e_sigma * lam + ((self.action - mean) ** 2 - sigma ** 2) * state
        self.w_sigma += self.stepsize_sigma * self.e_sigma * delta
        return mean, sigma

    def pick_action_from_dist(self, state):
        mean = np.dot(self.w_mu, state)
        sigma = np.exp(np.dot(self.w_sigma, state) + 0.000001)
        self.action = np.random.normal(mean, sigma)
        return self.action

