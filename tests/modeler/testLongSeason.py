import unittest
from pydlm.modeler.longSeason import longSeason


class testLongSeason(unittest.TestCase):


    def setUp(self):
        self.longSeason = longSeason(period=4, stay=4, w=1.0)
        self.longSeason2 = longSeason(period=2, stay=3, w=1.0)


    def testLongSeasonProperties(self):
        self.assertEqual(self.longSeason.componentType, 'longSeason')
        self.assertEqual(self.longSeason.period, 4)
        self.assertEqual(self.longSeason.stay, 4)

        self.assertEqual(self.longSeason2.componentType, 'longSeason')
        self.assertEqual(self.longSeason2.period, 2)
        self.assertEqual(self.longSeason2.stay, 3)

        
    def testUpdateEvaluation(self):
        trueFeatures = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                        [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                        [0, 0, 0, 1]]
        for i in range(13):
            self.longSeason.updateEvaluation(step=i)
            self.assertEqual(self.longSeason.evaluation.A1.tolist(),
                             trueFeatures[i])

    def testUpdateEvaluation2(self):
        trueFeatures = [[1, 0], [1, 0], [1, 0],
                        [0, 1], [0, 1], [0, 1],
                        [1, 0], [1, 0], [1, 0],
                        [0, 1], [0, 1], [0, 1],
                        ]
        for i in range(12):
            self.longSeason2.updateEvaluation(step=i)
            self.assertEqual(self.longSeason2.evaluation.A1.tolist(),
                             trueFeatures[i])



if __name__ == '__main__':
    unittest.main()
