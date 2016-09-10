import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.dlm import dlm

class testDlm(unittest.TestCase):

    def setUp(self):
        self.data = [0] * 9 + [1] + [0] * 10
        self.features = np.random.random((20, 2)).tolist()
        self.dlm1 = dlm(self.data)
        self.dlm2 = dlm(self.data)
        self.dlm3 = dlm([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        self.dlm1 + trend(degree = 1, discount = 1)
        self.dlm2 + trend(degree = 1, discount = 1e-12)
        self.dlm3 + seasonality(period = 2, discount = 1)

    def testFitForwardFilter(self):
        self.dlm1.fitForwardFilter(useRollingWindow = False)
        self.assertEqual(self.dlm1.result.filteredSteps, [0, 19])
        self.assertAlmostEqual(np.sum(self.dlm1.result.filteredObs[0:9]), 0)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[9], 1.0/11)
        self.assertAlmostEqual(self.dlm1.result.filteredObs[19], 1.0/21)

        self.dlm2.fitForwardFilter(useRollingWindow = False)
        self.assertAlmostEqual(np.sum(self.dlm2.result.filteredObs[0:9]), 0.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[9], 1.0)
        self.assertAlmostEqual(self.dlm2.result.filteredObs[19], 0.0)
        
    def testFitBackwardSmoother(self):
        self.dlm1.fitForwardFilter()
        self.dlm1.fitBackwardSmoother()
        self.assertEqual(self.dlm1.result.smoothedSteps, [0, 19])
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[0], 1.0/21)
        self.assertAlmostEqual(self.dlm1.result.smoothedObs[19], 1.0/21)

        self.dlm2.fitForwardFilter()
        self.dlm2.fitBackwardSmoother()
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[0], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[19], 0.0)
        self.assertAlmostEqual(self.dlm2.result.smoothedObs[9], 1.0)

    def testPredict(self):
        self.dlm3.fitForwardFilter()
        (obs, var) = self.dlm3._predict(days = 1)
        self.assertAlmostEqual(obs[0], -6.0/7)

    def testAppend(self):
        dlm4 = dlm(self.data[0:11])
        dlm4 + trend(degree = 1, discount = 1)
        dlm4.fitForwardFilter()
        self.assertEqual(dlm4.n, 11)

        dlm4.append(self.data[11 : 20])
        self.assertEqual(dlm4.n, 20)
        dlm4.fitForwardFilter()

        self.dlm1.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(self.dlm1.result.filteredObs)), 0.0)

    def testAppendDynamic(self):
        # we feed the data to dlm4 via two segments
        dlm4 = dlm(self.data[0:11])        
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = self.features[0:11], \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        dlm4.append(self.data[11 : 20])
        dlm4.append(self.features[11 : 20], component = 'dynamic')
        dlm4.fitForwardFilter()

        # we feed the data to dlm5 all at once
        dlm5 = dlm(self.data)
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = self.features, \
                                                         discount = 1)
        dlm5.fitForwardFilter()
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)       

    def testPopout(self):
        dlm4 = dlm(self.data)
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        # pop out the first date, the filtered range should be (0, -1)
        dlm4.popout(0)
        self.assertEqual(dlm4.result.filteredSteps, [0, -1])
        
        dlm4.fitForwardFilter()
        dlm5 = dlm(self.data[1 : 20])
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features[1 : 20], \
                                                         discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testAlter(self):
        dlm4 = dlm(self.data)
        dlm4 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm4.fitForwardFilter()
        # the filtered step range should be (0, 19)
        self.assertEqual(dlm4.result.filteredSteps, [0, 19])

        # pop out the first date, the filtered range should be (0, -1)
        dlm4.alter(date = 15, data = 1, component = 'main')
        self.assertEqual(dlm4.result.filteredSteps, [0, 14])
        dlm4.fitForwardFilter()

        newData = [0] * 9 + [1] + [0] * 10
        newData[15] = 1
        dlm5 = dlm(newData)
        dlm5 + trend(degree = 1, discount = 1) + dynamic(features = \
                                                         self.features, \
                                                         discount = 1)
        dlm5.fitForwardFilter()

        # The two chain should have the same filtered obs
        self.assertAlmostEqual(np.sum(np.array(dlm4.result.filteredObs) - \
                                      np.array(dlm5.result.filteredObs)), 0.0)

    def testTrunOn(self):
        pass

    def testTurnOff(self):
        pass

    def testSetColor(self):
        pass

    def testSetConfidence(self):
        pass

    
unittest.main()



#import numpy as np
#import pydlm as pd

#data = np.concatenate((np.random.random(100), np.random.random(100) + 3))
#myDLM = pd.dlm(data) + pd.trend(2, discount = 0.9)


#myDLM.turnOn('smooth')
#myDLM.turnOn('predict')
#myDLM.turnOff('multiple')
#myDLM.shrink(0.0)
#myDLM.fitForwardFilter(useRollingWindow = False, windowLength = 20)
#myDLM.fitForwardFilter()
#myDLM.fitBackwardSmoother()
#myDLM.plot()
