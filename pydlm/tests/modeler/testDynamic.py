import numpy as np
from pydlm.modeler.dynamic import dynamic
import unittest


class testDynamic(unittest.TestCase):

    def setUp(self):
        self.features = np.matrix(np.random.rand(10, 2)).tolist()
        self.newDynamic = dynamic(features=self.features, w=1.0)

    def testInputNumpyMatrix(self):
        dynamic(features=np.random.rand(10, 2), w=1.0)
        pass

    def testInitialization(self):
        self.assertEqual(self.newDynamic.d, 2)
        self.assertEqual(self.newDynamic.n, 10)
        self.assertAlmostEqual(np.abs(np.sum(np.matrix(
            self.newDynamic.features)
                - np.matrix(self.features))), 0)

    def testUpdate(self):
        self.newDynamic.updateEvaluation(3)
        self.assertAlmostEqual(np.abs(
            np.sum(np.array(self.newDynamic.evaluation)
                   - np.array(self.features[3]))), 0)

    def testAppendNewData(self):
        self.newDynamic.appendNewData([[1, 2]])
        self.assertAlmostEqual(np.abs(
            np.sum(np.array(self.newDynamic.features[-1])
                   - np.array([1, 2]))), 0)

    def testPopout(self):
        self.newDynamic.popout(0)
        self.assertAlmostEqual(np.abs(np.sum(
            np.matrix(self.newDynamic.features)
            - np.matrix(self.features[1:]))), 0)

    def testAlter(self):
        self.newDynamic.alter(1, [0, 0])
        self.assertAlmostEqual(self.newDynamic.features[1],
                               [0, 0])

if __name__ == '__main__':
    unittest.main()
