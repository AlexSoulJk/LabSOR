from collections.abc import Callable

import numpy as np


def linear_kernel_func(support_vec: np.ndarray, vec: np.ndarray) -> float:
    return support_vec.T * vec


def poly_kernel_func(support_vec: np.ndarray, vec: np.ndarray) -> float:
    return (1 + support_vec.T * vec) ** 2


def gauss_kernel_func(support_vec: np.ndarray, vec: np.ndarray) -> float:
    return np.exp(-(np.linalg.norm(support_vec - vec) ** 2) / 0.8)


def gauss_kernel_func_n(support_vec: np.ndarray, vec: np.ndarray) -> float:
    return np.exp(-(np.linalg.norm(support_vec - vec) ** 2) / 0.2)


class SorSvr(object):
    def __init__(self, epsilon: float = 0.1, sor_epsilon: float = 0.001,
                 omega: float = 0.5, C: int = 100, kernel: Callable = gauss_kernel_func):
        """
        Constructor

        :param epsilon: threshold for predicted function insensitivity
        :param sor_epsilon: threshold for SOR solver
        :param omega: magic const for SOR solver. Need to be in (0, 2)
        :param C: magic const for Lagrange function ¯\_(ツ)_/¯
        :param kernel: kernel function for "kernel trick"
        """
        self.epsilon = epsilon
        self.sor_epsilon = sor_epsilon
        self.C = C
        self.omega = omega
        self.kernel = kernel
        self.dim = 0
        self.support_vecs = []
        self.support_vecs_coeffs = []
        self.b = 0.0

    def _sor_solver(self, A, b, initial_guess):
        """
        This is an implementation of the pseudo-code provided in the Wikipedia article.
        Arguments:
            A: nxn numpy matrix.
            b: n dimensional numpy vector.
            initial_guess: An initial solution guess for the solver to start with.
        Returns:
            phi: solution vector of dimension n.
        """
        phi = initial_guess[:]
        residual = np.inf
        iter_count = 0
        while residual > self.sor_epsilon:
            iter_count += 1
            old_phi = phi.copy()
            for i in range(A.shape[0]):
                sigma = 0
                for j in range(A.shape[1]):
                    if j != i:
                        sigma += A[i][j] * phi[j]
                phi[i] = (1 - self.omega) * phi[i] + (self.omega / A[i][i]) * (b[i] - sigma)
                # phi[i] = phi[i] + (self.omega / A[i][i]) * (b[i] - sigma)
                if phi[i] < 0:
                    phi[i] = 0
                elif phi[i] > self.C:
                    phi[i] = self.C
            residual = np.linalg.norm(phi - old_phi)
            print('Residual: {0:10.6g}'.format(residual))
        print("\n~~~~~~ Iters count: {} ~~~~~~\n".format(iter_count))
        return phi

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.dim = len(x_train)
        # Create matrices from article yong2004
        x = x_train.reshape((self.dim, 1))
        y = y_train.reshape((self.dim, 1))
        d = np.ones((2 * self.dim, 1))
        d[self.dim:] *= -1
        c = np.full(shape=(2 * self.dim, 1), fill_value=-self.epsilon, dtype=np.float64)
        c[:self.dim] += y
        c[self.dim:] += -y
        H = d * d.T
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                x1 = x[i] if i < self.dim else x[i - self.dim]
                x2 = x[j] if j < self.dim else x[j - self.dim]
                H[i][j] *= self.kernel(x1, x2)
        E = d * d.T
        A = H + E
        # Find Lagrange multipliers by SOR algorithm
        a = self._sor_solver(A, c.reshape((2 * self.dim)), np.zeros(2 * self.dim))
        # Get support vectors and their coeffs (a_i - a*_i)
        self.support_vecs = []
        self.support_vecs_coeffs = []
        for i in range(0, self.dim):
            if not np.isclose(a[i], 0) or not np.isclose(a[self.dim + i], 0):
                self.support_vecs.append(x[i])
                self.support_vecs_coeffs.append(a[i] - a[self.dim + i])
        self.b = sum(self.support_vecs_coeffs)

    def predict(self, x: np.ndarray) -> float:
        res = 0.0
        for i in range(0, len(self.support_vecs)):
            res += self.support_vecs_coeffs[i] * self.kernel(self.support_vecs[i], x)
        return res + self.b

    def get_support_vecs(self):
        return self.support_vecs