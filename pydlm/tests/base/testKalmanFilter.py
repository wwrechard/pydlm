import numpy as np
import unittest

#sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

from pydlm.modeler.trends import trend
from pydlm.modeler.builder import builder
from pydlm.base.kalmanFilter import kalmanFilter

class testKalmanFilter(unittest.TestCase):

    def setUp(self):
        self.kf1 = kalmanFilter(discount = [1])
        self.kf0 = kalmanFilter(discount = [1e-10])
        
    def testForwardFilter(self):
        dlm = builder()
        dlm.add(trend(degree = 1, discount = 1))
        dlm.initialize()
        self.kf1.predict(dlm.model)
        self.assertAlmostEqual(dlm.model.prediction.obs, 0)

        # the prior on the mean is zero, but observe 1, with
        # discount = 1, one should expect the filterd mean to be 0.5
        self.kf1.forwardFilter(dlm.model, 1)
        self.assertAlmostEqual(dlm.model.obs, 0.5)
        self.assertAlmostEqual(dlm.model.prediction.obs, 0)
        self.assertAlmostEqual(dlm.model.sysVar, 0.625)

        self.kf1.predict(dlm.model)
        self.assertAlmostEqual(dlm.model.obs, 0.5)
        self.assertAlmostEqual(dlm.model.prediction.obs, 0.5)

        dlm.initialize()
        self.kf0.predict(dlm.model)
        self.assertAlmostEqual(dlm.model.prediction.obs, 0)

        # the prior on the mean is zero, but observe 1, with discount = 0
        # one should expect the filtered mean close to 1
        self.kf0.forwardFilter(dlm.model, 1)
        self.assertAlmostEqual(dlm.model.obs, 1)
        self.assertAlmostEqual(dlm.model.prediction.obs, 0)
        self.assertAlmostEqual(dlm.model.sysVar, 1)

        self.kf0.predict(dlm.model)
        self.assertAlmostEqual(dlm.model.obs, 1)
        self.assertAlmostEqual(dlm.model.prediction.obs, 1)


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
        self.assertAlmostEqual(dlm.model.obs, 1.0/3)
        self.assertAlmostEqual(dlm.model.sysVar, 0.43518519)
    
unittest.main()
#kf1 = kalmanFilter(discount = [1])
#kf0 = kalmanFilter(discount = [0.01])
#dlm = builder()
#dlm.add(trend(degree = 1, discount = 1))
#dlm.initialize()
#kf1.predict(dlm.model)
#kf1.forwardFilter(dlm.model, 1)
