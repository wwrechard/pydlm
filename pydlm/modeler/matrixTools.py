import numpy as np

## tools for concatenate matrix ###
class matrixTools:
    @staticmethod
    def matrixAddInDiag(A, B):
        if A is None:
            return np.matrix(B)
        elif B is None:
            return np.matrix(A)
        else:
            (An, Ap) = A.shape
            (Bn, Bp) = B.shape

            newMatrixA = np.concatenate((A, np.matrix(np.zeros((An, Bp)))), 1)
            newMatrixB = np.concatenate((np.matrix(np.zeros((Bn, Ap))), B), 1)
            return np.concatenate((newMatrixA, newMatrixB), 0)

    # A + B = (A; B)
    @staticmethod
    def matrixAddByRow(A, B):
        if A is None:
            return B
        elif B is None:
            return A
        else:
            return np.concatenate((A, B), 0)

    # A + B = (A B)
    @staticmethod
    def matrixAddByCol(A, B):
        if A is None:
            return np.matrix(B)
        elif B is None:
            return np.matrix(A)
        else:
            return np.concatenate((A, B), 1)

    @staticmethod
    def AddTwoVectors(a, b):
        if a is None:
            return np.array(b)
        elif b is None:
            return np.array(a)
        else:
            return np.concatenate((a, b))
