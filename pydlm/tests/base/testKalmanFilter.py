import numpy as np

#sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

from pydlm.modeler.trends import trend
from pydlm.modeler.builder import builder
from pydlm.base.kalmanFilter import kalmanFilter

class testKalmanFilter:

    def __init__(self):
        self.kf1 = kalmanFilter(discount = [1])
        self.kf0 = kalmanFilter(discount = [0.00000001])
        self.tol = 0.001

    def runTest(self):
        self.testForwardFilter()
        self.testBackwardSmoother()
        
    def testForwardFilter(self):
        dlm = builder()
        dlm.add(trend(degree = 1, discount = 1))
        dlm.initialize()
        self.kf1.predict(dlm.model)
        assert np.abs(dlm.model.prediction.obs - 0) < self.tol

        # the prior on the mean is zero, but observe 1, with
        # discount = 1, one should expect the filterd mean to be 0.5
        self.kf1.forwardFilter(dlm.model, 1)
        assert np.abs(dlm.model.obs - 0.5) < self.tol
        assert np.abs(dlm.model.prediction.obs - 0) < self.tol
        assert np.abs(dlm.model.sysVar - 0.625) < self.tol

        self.kf1.predict(dlm.model)
        assert np.abs(dlm.model.obs - 0.5) < self.tol
        assert np.abs(dlm.model.prediction.obs - 0.5) < self.tol

        dlm.initialize()
        self.kf0.predict(dlm.model)
        assert np.abs(dlm.model.prediction.obs - 0) < self.tol

        # the prior on the mean is zero, but observe 1, with discount = 0
        # one should expect the filtered mean close to 1
        self.kf0.forwardFilter(dlm.model, 1)
        assert np.abs(dlm.model.obs - 1) < self.tol
        assert np.abs(dlm.model.prediction.obs - 0) < self.tol
        assert np.abs(dlm.model.sysVar - 1) < self.tol

        self.kf0.predict(dlm.model)
        assert np.abs(dlm.model.obs - 1) < self.tol
        assert np.abs(dlm.model.prediction.obs - 1) < self.tol

        pass

    def testBackwardSmoother(self):
        dlm = builder()
        dlm.add(trend(degree = 1, discount = 1))
        dlm.initialize()

        # with mean being 0 and observe 1 and 0 consectively, one shall
        # expect the smoothed mean at 1 will be 1/3, for discount = 1
        self.kf1.forwardFilter(dlm.model, 1)
        self.kf1.forwardFilter(dlm.model, 0)
        self.kf1.backwardSmoother(dlm.model, \
                                  np.matrix([[0.5]]), \
                                  np.matrix([[0.625]]))
        assert np.abs(dlm.model.obs - 1.0/3) < self.tol
        assert np.abs(dlm.model.sysVar - 0.43518519) < self.tol

        pass
    
aTest = testKalmanFilter()
aTest.runTest()

#kf1 = kalmanFilter(discount = [1])
#kf0 = kalmanFilter(discount = [0.01])
#dlm = builder()
#dlm.add(trend(degree = 1, discount = 1))
#dlm.initialize()
#kf1.predict(dlm.model)
#kf1.forwardFilter(dlm.model, 1)
