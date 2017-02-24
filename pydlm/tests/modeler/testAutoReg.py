import unittest
from pydlm.modeler.autoReg import autoReg


class testAutoReg(unittest.TestCase):

    def setUp(self):
        data = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        self.ar4 = autoReg(degree=4, data=data, name='ar4', padding=0, w=1.0)

    def testFeatureMatrix(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 3)
        self.assertEqual(self.ar4.n, 12)

    def testAppendNewData(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3]]
        self.ar4.appendNewData([4])
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 4)
        self.assertEqual(self.ar4.n, 13)

    def testPopoutTheLast(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]
        self.ar4.popout(11)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 2)
        self.assertEqual(self.ar4.n, 11)

    def testPopoutInMiddel(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1],
                        [0, 0, 1, 3], [0, 1, 3, 0], [1, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        self.ar4.popout(2)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 3)
        self.assertEqual(self.ar4.n, 11)

    def testPopoutSecondLast(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]
        self.ar4.popout(10)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 3)
        self.assertEqual(self.ar4.n, 11)

    def testAlterTheLast(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        self.ar4.alter(11, 4)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 4)
        self.assertEqual(self.ar4.n, 12)

    def testAlterInMiddel(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 8],
                        [0, 1, 8, 3], [1, 8, 3, 0], [8, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
        self.ar4.alter(2, 8)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 3)
        self.assertEqual(self.ar4.n, 12)

    def testAlterSecondLast(self):
        trueFeatures = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 8]]
        self.ar4.alter(10, 8)
        self.assertEqual(self.ar4.features, trueFeatures)
        self.assertEqual(self.ar4.lastDay, 3)
        self.assertEqual(self.ar4.n, 12)

if __name__ == '__main__':
    unittest.main()
