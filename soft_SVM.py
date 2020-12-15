import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import param
from cvxopt import matrix, solvers
from kernel import Kernel


class SVM_predictor():
    def __init__(self,
                 kernel,
                 support_vector_langrange_multiplier,
                 support_vectors,
                 support_vector_labels,
                 bias
                 ):
        self._kernel = kernel
        self._support_vectors_langrange_multipliers = support_vector_langrange_multiplier
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        self._bias = bias

    def predict(self, x):
        return np.array([np.sign(self._bias + np.sum([y_i * a_i * self._kernel(x_i, x_j)
                                                      for y_i, a_i, x_i in zip(self._support_vector_labels,
                                                                               self._support_vectors_langrange_multipliers,
                                                                               self._support_vectors)])
                                 )
                         for x_j in x])

        # result = self._bias
        # for z_i, x_i, y_i in zip(self._support_vectors_langrange_multipliers,
        #                          self._support_vectors,
        #                          self._support_vector_labels):
        #     result += z_i * y_i * self._kernel(x_i, x)
        # return np.sign(result).item()

class SVM_trainer1():
    def __init__(self, kernel, c):
        """

        :param kernel:
        :param c: 误差惩罚系数，c越小，越不容易过拟合
     """
        self._kernel = kernel
        self._c = c
        self._MIN_SUPPORT_VECTOR_LAGRANGE_MULTIPLIER = param.MIN_SUPPORT_VECTOR_LAGRANGE_MULTIPLIER
        self._support_vectors = None

    def train(self, x, y):
        """

        :param x:sample
        :param y: label
        :return: trained SVM model
        """
        lagrange_multiplier = self._compute_lagrange_multiplier(x, y)
        return self._construct_SVM_model(x, y, lagrange_multiplier)

    def _x_matrix_generator(self, x):
        """

        :param x:sample
        :return: x_matrix whose c(ij) equals kernel(xi,xj)
        """
        x_matrix = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                x_matrix[i, j] = self._kernel(x[i], x[j])
        return x_matrix

    def _compute_lagrange_multiplier(self, x, y):
        n_samples, n_features = x.shape
        x_matrix = self._x_matrix_generator(x)

        """
        qp solver:
            min 1/2 x^T P x + q^T x
            s.t.
            Gx <= h
            Ax = b
        """

        P = matrix(np.outer(y, y) * x_matrix)
        q = matrix(-1 * np.ones(n_samples))

        G_top = -1 * np.diag(np.ones(n_samples))
        h_top = matrix(np.zeros(n_samples))

        G_down = np.diag(np.ones(n_samples))
        h_down = matrix(np.ones(n_samples) * self._c)

        G = matrix(np.vstack([G_top, G_down]))
        h = matrix(np.vstack([h_top, h_down]))

        A = matrix(y, (1, n_samples))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

    def _construct_SVM_model(self, x, y, lagrange_multiplier):
        support_vector_indices = lagrange_multiplier > self._MIN_SUPPORT_VECTOR_LAGRANGE_MULTIPLIER
        support_vector_lagrange_multiplier = lagrange_multiplier[support_vector_indices]
        self._support_vectors = x[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        """
        due to the accuracy of calculation,
         the value of bias is not sole,
         hence we represent bias by it's mean
        """

        svm_predictor1 = SVM_predictor(self._kernel,
                                       support_vector_lagrange_multiplier,
                                       self._support_vectors,
                                       support_vector_labels,
                                       bias=0.0)

        bias = np.mean(
            [y_i - svm_predictor1.predict(x_i)
             for y_i, x_i in zip(support_vector_labels, self._support_vectors)]
        )

        return SVM_predictor(
            self._kernel,
            support_vector_lagrange_multiplier,
            self._support_vectors,
            support_vector_labels,
            bias=bias)