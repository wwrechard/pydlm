# this class provide all the model building operations for constructing customized model
import numpy as np
from pydlm.base.baseModel import baseModel
import matrixTools as mt

# The builder will be the main class for construting dlm
# it featues two types of evaluation matrix and evaluation matrix
class builder:

    # create components
    def __init__(self):

        # the basic model structure for running kalman filter
        self.model = baseModel()
        self.initialized = False

        # to store all components. Separate the two as the evaluation
        # for dynamic component needs update each iteration
        self.staticComponents = []
        self.dynamicComponents = []

        # to form two evaluation matrices
        self.staticEvaluation = None
        self.dynamicEvaluation = None

        # record the current step/days/time stamp
        self.step = 0

    def add(self, component):
        if component.dynamic:
            self.dynamicComponents.append(component)
        else:
            self.staticComponents.append(component)

    # print all components to the client
    def listComponents(self):
        if len(self.staticComponents) > 0:
            print 'The static components are'
            for compIdx in range(len(self.staticComponents)):
                comp = self.staticComponents[compIdx]
                print comp.name + ' of' + str(comp.d) + '. Index: ' + str(compIdx)
        else:
            print 'There is no static component.'

        if len(self.dynamicComponents) > 0:
            print 'The dynamic components are'
            for compIdx in range(len(self.dynamicComponents)):
                comp = self.dynamicComponents[compIdx]
                print comp.name + ' of' + str(comp.d) + '. Index: ' + \
                    str(compIdx + len(self.staticComponents))
        else:
            print 'There is no dynamic component.'

    # print delete some componets
    def delete(self, index):
        if index < 0 or index >= len(self.staticComponents) + len(self.dynamicComponents):
            raise NameError('The index is out of range.')
        elif index < len(self.staticComponents):
            self.staticComponents.pop(index)
        else:
            self.dynamicComponents.pop(index - len(self.staticComponents))
        self.initialized = False

    # initializing model
