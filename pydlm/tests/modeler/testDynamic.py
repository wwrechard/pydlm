import numpy as np
import pydlm
class testDynamic:

    def __init__(self):
        self.features = np.matrix(np.random.rand(2, 10))
        self.newDynamic = pydlm.modeler.dynamic.dynamic(features = self.features)
        self.tol = 0.001

    def runTest(self):
        self.testInitialization()
        self.testUpdate()
        
    def testInitialization(self):        
        assert self.newDynamic.d == 2
        assert self.newDynamic.n == 10
        assert np.sum(np.abs(self.newDynamic.evaluation - \
                             self.features[:, 0].T)) < self.tol
        pass

    def testUpdate(self):
        self.newDynamic.updateEvaluation(3)
        assert np.sum(np.abs(self.newDynamic.evaluation - \
                             self.features[:, 3].T)) < self.tol
        pass

aTest = testDynamic()
aTest.runTest()
