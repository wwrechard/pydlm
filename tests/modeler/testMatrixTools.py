import numpy as np
import unittest

from pydlm.modeler.matrixTools import matrixTools as mt

class testMatrixTools(unittest.TestCase):

    def testMatrixAddByRow(self):
        A = np.array([[1], [2]])
        B = np.array([[3]])
        expected = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(mt.matrixAddByRow(A, B), expected)

        A = np.array([1, 2])
        B = np.array([3])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(mt.matrixAddByRow(A, B), expected)

    def testMatrixAddByCol(self):
        A = np.array([[1]])
        B = np.array([[3]])
        expected = np.array([[1, 3]])
        np.testing.assert_array_equal(mt.matrixAddByCol(A, B), expected)

        A = np.array([[1], [2]])
        B = np.array([[3], [4]])
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(mt.matrixAddByCol(A, B), expected)

        
if __name__ == '__main__':
    unittest.main()
