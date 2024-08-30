import unittest
from pydlm.modeler.autoReg import autoReg


class testAutoReg(unittest.TestCase):


    def setUp(self):
        self.data = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        self.ar4 = autoReg(degree=4, name='ar4', padding=0, w=1.0)


    def testFeatureMatrix(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        for i in range(12):
            self.ar4.updateEvaluation(i, self.data)
            self.assertEqual(self.ar4.evaluation.flatten().tolist(), trueFeatures[i])


if __name__ == '__main__':
    unittest.main()
