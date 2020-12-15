import numpy as np

class Kernel():
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)

        return f

    @staticmethod
    def polykernel(dim, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dim

        return f

    @staticmethod
    def inhomogeneous_polynominal(dim):
        return Kernel.polykernel(dim=dim, offset=1.0)

    @staticmethod
    def homogeneous_polynominal(dim):
        return Kernel.polykernel(dim=dim, offset=0)

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2))
            return np.exp(exponent)

        return f

    @staticmethod
    def heperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.inner(x, y) + c)

        return f