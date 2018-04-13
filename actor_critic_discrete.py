import numpy as np
from utils import Q_feature_vector

class ActorCriticDiscrete:

    def __init__(self, num_state, num_actions, stepsize_v, stepsize_u, stepsize_r):

        # step sizes
        self.stepsize_u = stepsize_u
        self.stepsize_v = stepsize_v
        self.stepsize_r = stepsize_r

        # num of features and actions and list of possible actions
        self.num_state = num_state
        self.num_actions = num_actions
        self.possible_action_inds = [0, 1]

        # average reward
        self.R_bar = 0.0

        # elegibility traces
        self.e_v = np.zeros((self.num_state))
        self.e_u = np.zeros((self.num_state * self.num_actions))

        # weights
        self.w_v = np.zeros((self.num_state))
        self.w_u = np.zeros((self.num_state * self.num_actions))

        # latest action
        self.action = None

    def update(self, state, reward, state_prime, lam):

        delta = reward - self.R_bar + np.dot(self.w_v, state_prime) - np.dot(self.w_v, state)
        self.critic_step(state, lam, delta)
        self.actor_step(state, lam, delta)

    def critic_step(self, state, lam, delta):

        self.R_bar += self.stepsize_r * delta
        self.e_v = lam * self.e_v + state
        self.w_v += self.stepsize_v * delta * self.e_v

    def actor_step(self, state, lam, delta):

        grad_log_prob = np.zeros(self.num_state * len(self.possible_action_inds))
        for action in range(len(self.possible_action_inds)):
            grad_log_prob += np.dot(self.get_action_prob(state, action), Q_feature_vector(state, action))

        self.e_u = lam * self.e_u + Q_feature_vector(state, self.action) - grad_log_prob
        self.w_u += self.stepsize_u * delta * self.e_u

    def get_action_prob(self, state, action):

        return np.exp(np.dot(self.w_u, Q_feature_vector(state, action))) / self.get_softmax_sum_of_all_actions(state)

    def get_softmax_sum_of_all_actions(self, state):

        sum = 0.0
        for action in range(len(self.possible_action_inds)):
            sum += np.exp(np.dot(self.w_u, Q_feature_vector(state, action)))
        return sum

    def get_softmax_actions(self, state):

        action_ind_prob = np.zeros(len(self.possible_action_inds))

        for action in range(len(self.possible_action_inds)):
            action_ind_prob[action] = self.get_action_prob(state, action)

        return action_ind_prob

    def pick_action_from_dist(self, state):

        action_ind_prob = self.get_softmax_actions(state)
        action = np.random.choice(len(self.possible_action_inds), p = action_ind_prob)
        self.action = action
        return action, action_ind_prob
