import math

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


# inverse normal cdf function
def rational_approximation(t):
 
    # Abramowitz and Stegun formula 26.2.23.
    # The absolute value of the error should be less than 4.5 e-4.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    numerator = (c[2]*t + c[1])*t + c[0]
    denominator = ((d[2]*t + d[1])*t + d[0])*t + 1.0
    return t - numerator / denominator
 
 
def normal_CDF_inverse(p):
    assert p > 0.0 and p < 1
 
    # See article above for explanation of this section.
    if p < 0.5:
        # F^-1(p) = - G^-1(p)
        return -rational_approximation( math.sqrt(-2.0*math.log(p)) )
    else:
        # F^-1(p) = G^-1(1-p)
        return rational_approximation( math.sqrt(-2.0*math.log(1.0-p)) )


def getInterval(means, var, p):
    alpha = abs(normal_CDF_inverse(min(1 - p, p) / 2))
    upper = [0] * len(means)
    lower = [0] * len(means)
    for i in range(0, len(means)):
        upper[i] = means[i] + alpha * math.sqrt(var[i])
        lower[i] = means[i] - alpha * math.sqrt(var[i])

    return (upper, lower)
