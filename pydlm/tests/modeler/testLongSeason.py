import unittest
from pydlm.modeler.longSeason import longSeason


class testLongSeason(unittest.TestCase):

    def setUp(self):
        data = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        self.longSeason = longSeason(data=data, period=4, stay=4, w=1.0)

    def testFeatureMatrix(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        self.assertEqual(self.longSeason.features, trueFeatures)
        self.assertEqual(self.longSeason.nextState, [3, 0])
        self.assertEqual(self.longSeason.n, 12)

    def testAppendNewData(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                        [0, 0, 0, 1]]
        self.longSeason.appendNewData([4])
        self.assertEqual(self.longSeason.features, trueFeatures)
        self.assertEqual(self.longSeason.nextState, [3, 1])
        self.assertEqual(self.longSeason.n, 13)

    def testPopoutTheLast(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        self.longSeason.popout(11)
        self.assertEqual(self.longSeason.features, trueFeatures)
        self.assertEqual(self.longSeason.nextState, [2, 3])
        self.assertEqual(self.longSeason.n, 11)

    def testPopoutInMiddel(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        self.longSeason.popout(1)
        self.assertEqual(self.longSeason.features, trueFeatures)
        self.assertEqual(self.longSeason.nextState, [2, 3])
        self.assertEqual(self.longSeason.n, 11)

    def testUpdateEvaluation(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                        [0, 0, 0, 1]]
        self.longSeason.updateEvaluation(12)
        self.assertEqual(self.longSeason.features, trueFeatures)
        self.assertEqual(self.longSeason.nextState, [3, 1])
        self.assertEqual(self.longSeason.n, 12)

if __name__ == '__main__':
    unittest.main()
