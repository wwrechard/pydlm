# this class provide all the model building operations for constructing customized model
import numpy as np
from pydlm.base.baseModel import baseModel
from matrixTools import matrixTools as mt

# The builder will be the main class for construting dlm
# it featues two types of evaluation matrix and evaluation matrix
# The static evaluation remains the same over time which is used to
# record the trend and seasonality.
#
# The dynamic evaluation vector changes over time, it is basically
# the other variables that might have impact on the time series
# We need to update this vector as time going forward
class builder:

    # create members
    def __init__(self):

        # the basic model structure for running kalman filter
        self.model = None
        self.initialized = False

        # to store all components. Separate the two as the evaluation
        # for dynamic component needs update each iteration
        self.staticComponents = {}
        self.dynamicComponents = {}
        self.componentIndex = {}

        # record the prior guess on the latent state and system covariance
        self.statePrior = None
        self.sysVarPrior = None
        self.noiseVar = None
        
        # record the current step/days/time stamp
        self.step = 0

        # record the discount factor for the model
        self.discount = None


    # The function that allows the user to add components
    def add(self, component):
        self.__add__(component)
        
    def __add__(self, component):
        if component.dynamic:
            if component.name in self.dynamicComponents:
                raise NameError('Please rename the component to a different name.')
            self.dynamicComponents[component.name] = component
        else:
            if component.name in self.staticComponents:
                raise NameError('Please rename the component to a different name.')
            self.staticComponents[component.name] = component
        self.initialized = False
        return self

    # print all components to the client
    def ls(self):
        if len(self.staticComponents) > 0:
            print 'The static components are'
            for name in self.staticComponents:
                comp = self.staticComponents[name]
                print comp.name + ' (degree = ' + str(comp.d) + ')'
            print ' '
        else:
            print 'There is no static component.'
            print ' '

        if len(self.dynamicComponents) > 0:
            print 'The dynamic components are'
            for name in self.dynamicComponents:
                comp = self.dynamicComponents[name]
                print comp.name + ' (dimension = ' + str(comp.d) + ')'
        else:
            print 'There is no dynamic component.'

    # delete the componet that pointed out by the client
    def delete(self, name):
        if name in self.staticComponents:
            del self.staticComponents[name]
        elif name in self.dynamicComponents:
            del self.dynamicComponents[name]
        else:
            raise NameError('Such component does not exisit!')
            
        self.initialized = False

    # initialize model for all the quantities
    # noise is the prior guess of the variance of the observed data
    def initialize(self, noise = 1):
        if len(self.staticComponents) == 0:
            raise NameError('The model must contain at least one static component')
        
        # construct transition, evaluation, prior state, prior covariance
        print 'Constructing the basic quantities...'
        transition = None
        evaluation = None
        state = None
        sysVar = None
        self.discount = np.array([])

        # first construct for the static components
        # the evaluation will be treated separately for static or dynamic
        # as the latter one will change over time
        currentIndex = 0 # used for compute the index
        for i in self.staticComponents:
            comp = self.staticComponents[i]
            transition = mt.matrixAddInDiag(transition, comp.transition)
            evaluation = mt.matrixAddByCol(evaluation, \
                                           comp.evaluation)
            state = mt.matrixAddByRow(state, comp.meanPrior)
            sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)
            self.discount = np.concatenate((self.discount, comp.discount))
            self.componentIndex[i] = (currentIndex, currentIndex + comp.d - 1)
            currentIndex += comp.d

        # if the model contains the dynamic part, we add the dynamic components
        if len(self.dynamicComponents) > 0:
            self.dynamicEvaluation = None
            for i in self.dynamicComponents:
                comp = self.dynamicComponents[i]
                transition = mt.matrixAddInDiag(transition, comp.transition)
                evaluation = mt.matrixAddByCol(evaluation, \
                                               comp.evaluation)
                state = mt.matrixAddByRow(state, comp.meanPrior)
                sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)
                self.discount = np.concatenate((self.discount, comp.discount))
                self.componentIndex[i] = (currentIndex, currentIndex + comp.d - 1)
                currentIndex += comp.d
        
        print 'Writing to the base model...'
        self.statePrior = state
        self.sysVarPrior = sysVar
        self.noiseVar = np.matrix(noise)
        self.model = baseModel(transition = transition, \
                               evaluation = evaluation, \
                               noiseVar = np.matrix(noise), \
                               sysVar = sysVar, \
                               state = state, \
                               df = 1)
        self.model.initializeObservation()
        self.initialized = True
        print 'Initialization finished.'

    # This function allows the model to update the dynamic evaluation vector,
    # so that the model can handle control variables
    # This function should be called only when dynamicComponents is not empty
    def updateEvaluation(self, step = None):
        if len(self.dynamicComponents) == 0:
            raise NameError('This shall only be used when there are dynamic components!')

        # obtain the correct step
        if step is None:
            self.step += 1
            step = self.step
        else:
            self.step = step

        # update the dynamic evaluation vector
        # We need first update all dynamic components by 1 step
        for i in self.dynamicComponents:
            comp = self.dynamicComponents[i]
            comp.updateEvaluation(step)
            self.model.evaluation[0, self.componentIndex[i][0] : \
                                  (self.componentIndex[i][1] + 1)] = comp.evaluation
