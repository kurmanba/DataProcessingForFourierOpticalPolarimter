from collections import defaultdict
import numpy as np


class MullerOperators:

    def __init__(self, teta, sigma):

        self.teta = teta
        self.sigma = sigma

    def wave_pallet(self):

        r_0 = self.rotation_matrix(self, sign=1)
        r_1 = self.rotation_matrix(self, sign=-1)
        matrix = self.muller_matrix('LP')

        return r_0 @ matrix @ r_1

    def rotation_matrix(self,
                        sign: any) -> np.ndarray:

        a = np.cos(2 * self.teta * sign)
        b = np.sin(2 * self.teta * sign)

        rotate = np.array([[1, 0, 0, 0],
                           [1, a, b, 0],
                           [0, -b, a, 0],
                           [0, 0, 0, 1]], np.float64)
        return rotate

    @staticmethod
    def muller_matrix(operator: str) -> defaultdict:

        muller = defaultdict()
        muller['LP'] = np.array([[1, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], np.float64)

        muller['LP'] = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], np.float64)

        return muller[operator]


z = MullerOperators
z.wave_pallet()
