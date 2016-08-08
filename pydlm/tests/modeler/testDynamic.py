import numpy as np
import pydlm
import unittest

class testDynamic(unittest.TestCase):

    def setUp(self):
        self.features = np.matrix(np.random.rand(2, 10))
        self.newDynamic = pydlm.modeler.dynamic.dynamic(features = self.features)
        
    def testInitialization(self):        
        self.assertEqual(self.newDynamic.d, 2)
        self.assertEqual(self.newDynamic.n, 10)
        self.assertEqual(np.sum(np.abs(self.newDynamic.evaluation - \
                                       self.features[:, 0].T)), 0)

    def testUpdate(self):
        self.newDynamic.updateEvaluation(3)
        self.assertEqual(np.sum(np.abs(self.newDynamic.evaluation - \
                                       self.features[:, 3].T)), 0)

unittest.main()
