import numpy as np
from pydlm.modeler.dynamic import dynamic
import unittest


class testDynamic(unittest.TestCase):


    def setUp(self):
        self.features = np.matrix(np.random.rand(10, 2)).tolist()
        self.features2 = np.matrix(np.random.rand(10, 1)).tolist()
        self.newDynamic = dynamic(features=self.features, w=1.0)
        self.newDynamic2 = dynamic(features=self.features2, w=1.0)


    def testInputNumpyMatrix(self):
        dynamic(features=np.random.rand(10, 2), w=1.0)
        pass


    def testInitialization(self):
        self.assertEqual(self.newDynamic.d, 2)
        self.assertEqual(self.newDynamic.n, 10)
        self.assertEqual(self.newDynamic.features, self.features)

        self.assertEqual(self.newDynamic2.d, 1)
        self.assertEqual(self.newDynamic2.n, 10)
        self.assertEqual(self.newDynamic2.features, self.features2)



    def testUpdate(self):
        for i in range(10):
            self.newDynamic.updateEvaluation(i)
            np.testing.assert_array_equal(
                self.newDynamic.evaluation, np.matrix([self.features[i]]))

        for i in range(10):
            self.newDynamic2.updateEvaluation(i)
            np.testing.assert_array_equal(
                self.newDynamic2.evaluation, np.matrix([self.features2[i]]))


    def testAppendNewData(self):
        self.newDynamic.appendNewData([[1, 2]])
        self.assertEqual(self.newDynamic.features[-1], [1, 2])


    def testPopout(self):
        self.newDynamic.popout(0)
        self.assertEqual(self.newDynamic.features, self.features[1:])


    def testAlter(self):
        self.newDynamic.alter(1, [0, 0])
        self.assertEqual(self.newDynamic.features[1], [0, 0])


if __name__ == '__main__':
    unittest.main()
