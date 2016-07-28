import sys
import numpy as np

sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

import pydlm
from pydlm.modeler.matrixTools import matrixTools as mt
class testBuilder:

    def __init__(self):
        self.features = np.matrix(np.random.rand(2, 10))
        self.trendD = 3
        self.seasonalityD = 7
        self.tol = 0.001

        self.trend = pydlm.modeler.trends.trend(self.trendD)
        self.seasonality = pydlm.modeler.seasonality.seasonality(self.seasonalityD)
        self.dynamic = pydlm.modeler.dynamic.dynamic(self.features)

    def testRun(self):
        self.testInitialization()
        self.testAddAndDelete()
        self.testInitialize()
        self.testUpdate()
        
    def testInitialization(self):
        builder = pydlm.modeler.builder.builder()
        assert builder.step == 0
        assert len(builder.dynamicComponents) == 0
        pass

    def testAddAndDelete(self):
        builder = pydlm.modeler.builder.builder()
        builder.add(self.trend)
        assert len(builder.staticComponents) == 1
        assert len(builder.dynamicComponnets) == 0

        builder.add(self.dynamic)
        assert len(builder.staticComponents) == 1
        assert len(builder.dynamicComponnets) == 1

        builder.add(self.seasonality)
        assert len(builder.staticComponents) == 2
        assert len(builder.dynamicComponnets) == 1

        builder.delete(1)
        assert len(builder.staticComponents) == 1
        assert len(builder.dynamicComponnets) == 1

        assert builder.staticComponents[0] == self.trend
        pass
    
    def testInitialize(self):
        builder = pydlm.modeler.builder.builder()
        builder.add(self.trend)
        builder.add(self.dynamic)
        
        builder.initialize()
        assert np.sum(np.abs(builder.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, self.dynamic.evaluation))) < self.tol
        pass

    def testUpdate(self):
        builder = pydlm.modeler.builder.builder()
        builder.add(self.trend)
        builder.add(self.dynamic)        
        builder.initialize()

        builder.updateEvaluation(2)
        assert np.sum(np.abs(builder.model.evaluation - mt.matrixAddByCol(self.trend.evaluation, self.features[:, 2].T))) < self.tol
        pass
