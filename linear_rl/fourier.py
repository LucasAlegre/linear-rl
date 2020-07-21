import numpy as np
from itertools import combinations

from linear_rl.basis import Basis


class FourierBasis(Basis):

    def __init__(self, state_space, action_space, order):
        super().__init__(state_space, action_space)
        self.order = order
        self.coeff = self._build_coefficients()
    
    def get_learning_rates(self, alpha):
        lrs = np.linalg.norm(self.coeff, axis=1)
        lrs[lrs==0.] = 1.
        lrs = alpha/lrs
        return lrs
    
    def _build_coefficients(self):
        coeff = np.array(np.zeros(self.state_dim))  # Bias
        for order in range(1, self.order + 1):
            coeff = np.vstack((coeff, np.identity(self.state_dim)*order))
        for i, j in combinations(range(self.state_dim), 2):
            for o1 in range(1, self.order + 1):
                for o2 in range(1, self.order + 1):
                    c = np.zeros(self.state_dim)
                    c[i] = o1
                    c[j] = o2
                    coeff = np.vstack((coeff, c))
        return coeff
    
    def get_features(self, state):
        # scale to [0,1]
        state = (state - self.state_space.low) / (self.state_space.high - self.state_space.low)
        return np.cos(np.dot(np.pi*self.coeff, state))

    def get_num_basis(self) -> int:
        return len(self.coeff)