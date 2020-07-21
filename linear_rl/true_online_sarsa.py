import numpy as np
from linear_rl.fourier import FourierBasis


class TrueOnlineSarsaLambda:

    def __init__(self, state_space, action_space, basis='fourier', alpha=0.001, lamb=0.9, gamma=0.99, epsilon=0.05, fourier_order=7):
        self.alpha = alpha
        self.lr = alpha
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = epsilon

        self.state_space = state_space
        self.state_dim = self.state_space.shape[0]
        self.action_space = action_space
        self.action_dim = self.action_space.n

        if basis == 'fourier':
            self.basis = FourierBasis(self.state_space, self.action_space, fourier_order)
            self.lr = self.basis.get_learning_rates(self.alpha)

        self.num_basis = self.basis.get_num_basis()

        self.et = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}
        self.theta = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}

        self.q_old = None
        self.action = None

    def learn(self, state, action, reward, next_state, done):
        phi = self.get_features(state)
        next_phi = self.get_features(next_state)
        q = self.get_q_value(phi, action)
        next_q = self.get_q_value(next_phi, self.get_action(next_phi))
        td_error = reward + self.gamma * next_q - q
        if self.q_old is None:
            self.q_old = q

        for a in range(self.action_dim):
            if a == action:
                self.et[a] = self.lamb*self.gamma*self.et[a] + phi - self.lr*self.gamma*self.lamb*(self.et[a]*phi)*phi
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a] - self.lr*(q - self.q_old)*phi
            else:
                self.et[a] = self.lamb*self.gamma*self.et[a]
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a]
        
        self.q_old = next_q
        if done:
            self.reset_traces()

    def get_q_value(self, features, action):
        return np.dot(self.theta[action], features)
        
    def get_features(self, state):
        return self.basis.get_features(state)
    
    def reset_traces(self):
        self.q_old = None
        for a in range(self.action_dim):
            self.et[a].fill(0.0)
    
    def act(self, obs):
        features = self.get_features(obs)
        return self.get_action(features)

    def get_action(self, features):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.get_q_value(features, a) for a in range(self.action_dim)]
            return q_values.index(max(q_values))