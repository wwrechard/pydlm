# define the error class for exceptions
class matrixErrors(Exception):
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# The class to check matrixErrors
class checker:

    # checking if two matrix has the same dimension
    @staticmethod
    def checkMatrixDimension(A, B):
        if A.shape != B.shape:
            raise matrixErrors('The dimensions do not match!')

    # checking if a vector has the dimension as matrix
    @staticmethod
    def checkVectorDimension(v, A):
        vDim = v.shape
        ADim = A.shape
        if vDim[0] != ADim[0] and vDim[0] != ADim[1] and \
           vDim[1] != ADim[0] and vDim[1] != ADim[1]:
            raise matrixErrors('The dimensions do not match!')

    # checking if a matrix is symmetric
    @staticmethod
    def checkSymmetry(A):
        ADim = A.shape
        if ADim[0] != ADim[1]:
            raise matrixErrors('The matrix is not symmetric!')

# a completely unshared copy of lists
def duplicateList(aList):
    if isinstance(aList, list):
        return list(map(duplicateList, aList))
    return aList
