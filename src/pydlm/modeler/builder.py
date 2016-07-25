# this class provide all the model building operations for constructing customized model
import numpy as np
from pydlm.base.baseModel import baseModel

class builder:

    # create components
    def __init__(self):

        self.model = baseModel()
        self.initialized = False

        # to remember the indexes of each component
        self.staticComponents = []
        self.dynamicComponents = []

        # to form two transition matrices
        self.staticTransition = None
        self.dynamicTransition = None

        # record the current step/days/time stamp
        self.step = 0
