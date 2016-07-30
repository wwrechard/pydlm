import sys
import numpy as np

sys.path.append('/Users/samuel/Documents/Github/PyDLM/src/')

from pydlm.modeler.trends import trend
from pydlm.modeler.seasonality import seasonality
from pydlm.modeler.builder import builder
from pydlm.base.kalmanFilter import kalmanFilter

class testKalmanFilter:
    
