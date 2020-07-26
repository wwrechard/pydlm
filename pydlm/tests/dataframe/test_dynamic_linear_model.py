import unittest

import numpy as np
import pandas as pd
from numpy import array_equal as ae

from pydlm import trend, seasonality, dynamic
from pydlm.dataframe.dynamic_linear_model import DynamicLinearModel as d1
from pydlm.dlm import dlm as d2


class TestDynamicLinearModel(unittest.TestCase):

    def setUp(self):
        data = pd.DataFrame({'y': np.random.random(1000)})
        # construct the dlm of a linear trend and a 7-day seasonality
        m1 = d1(data, target_col='y') + trend(2, 0.98) + seasonality(7, 0.98)
        m1.fitForwardFilter()
        m1.fitBackwardSmoother()

        m2 = d2(data.y.values) + trend(2, 0.98) + seasonality(7, 0.98)
        m2.fitForwardFilter()
        m2.fitBackwardSmoother()

        self.m1 = m1
        self.m2 = m2

        data2 = pd.DataFrame({'y': np.random.random(1000),
                              'x1': np.random.random(1000),
                              'x2': np.random.random(1000)})

        m3 = d1(data2, target_col='y', x_cols=['x1', 'x2']) + trend(2, 0.98) + seasonality(7, 0.98)
        m3.fitForwardFilter()
        m3.fitBackwardSmoother()

        m4 = d2(data2.y.values) + trend(2, 0.98) + seasonality(7, 0.98) + \
             dynamic(features=data2.loc[:, ['x1', 'x2']].values)
        m4.fitForwardFilter()
        m4.fitBackwardSmoother()

        self.m3 = m3
        self.m4 = m4

    def testGets(self):
        self.assertEqual(self.m1.getMSE(), self.m2.getMSE())

        for filter_type in ['forwardFilter', 'backwardSmoother', 'predict']:
            self.assertTrue(
                ae(self.m1.getInterval(filterType=filter_type),
                   self.m2.getInterval(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m1.getMean(filterType=filter_type),
                   self.m2.getMean(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m1.getVar(filterType=filter_type),
                   self.m2.getVar(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m1.getLatentCov(filterType=filter_type),
                   self.m2.getLatentCov(filterType=filter_type))
            )

            self.assertTrue(
                ae(self.m3.getInterval(filterType=filter_type),
                   self.m4.getInterval(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m3.getMean(filterType=filter_type),
                   self.m4.getMean(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m3.getVar(filterType=filter_type),
                   self.m4.getVar(filterType=filter_type))
            )
            self.assertTrue(
                ae(self.m3.getLatentCov(filterType=filter_type),
                   self.m4.getLatentCov(filterType=filter_type))
            )
