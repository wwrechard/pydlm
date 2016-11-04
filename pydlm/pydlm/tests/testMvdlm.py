import numpy as np
import unittest

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.dynamic import dynamic
from pydlm.modeler.autoReg import autoReg
from pydlm.mvdlm import mvdlm

class testMvdlm(unittest.TestCase):

    def setUp(self):
        self.data = 
