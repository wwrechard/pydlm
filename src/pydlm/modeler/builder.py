# this class provide all the model building operations for constructing customized model
from pydlm.base.baseModel import baseModel
import matrixTools as mt

# The builder will be the main class for construting dlm
# it featues two types of evaluation matrix and evaluation matrix
class builder:

    # create components
    def __init__(self):

        # the basic model structure for running kalman filter
        self.model = None
        self.initialized = False

        # to store all components. Separate the two as the evaluation
        # for dynamic component needs update each iteration
        self.staticComponents = []
        self.dynamicComponents = []

        # record the evaluation vector for static or dynamic component
        # avoid recompute the static part
        self.staticEvaluation = None
        self.dynamicEvaluation = None

        # record the current step/days/time stamp
        self.step = 0

        # record the discount factor for the model
        self.discount = None


    # The function that allows the user to add components
    def add(self, component):
        if component.dynamic:
            self.dynamicComponents.append(component)
        else:
            self.staticComponents.append(component)

    # print all components to the client
    def listup(self):
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

    # delete the componet that pointed out by the client
    def delete(self, index):
        if index < 0 or index >= len(self.staticComponents) + len(self.dynamicComponents):
            raise NameError('The index is out of range.')
        elif index < len(self.staticComponents):
            self.staticComponents.pop(index)
        else:
            self.dynamicComponents.pop(index - len(self.staticComponents))
        self.initialized = False

    # initialize model for all the quantities
    # noise is the prior guess of the variance of the observed data
    def initialize(self, noise = 1):
        if len(self.staticComponents) == 0:
            raise NameError('The model must contain at least one static component')
        
        # construct transition, evaluation, prior state, prior covariance
        print 'Constructing the basic quantities...'
        transition = self.staticComponents[0].transition
        self.staticEvaluation = self.staticComponents[0].evaluation
        state = self.staticComponents[0].meanPrior
        sysVar = self.staticComponents[0].covPrior

        # first construct for the static components
        # the evaluation will be treated separately for static or dynamic
        # as the latter one will change over time
        for i in range(1, len(self.staticComponents)):
            comp = self.staticComponents[i]
            transition = mt.matrixAddInDiag(transition, comp.transition)
            self.staticEvaluation = mt.matrixAddByCol(self.staticEvaluation, \
                                                      comp.evaluation)
            state = mt.matrixAddByRow(state, comp.meanPrior)
            sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)

        # if the model contains the dynamic part, we add the dynamic components
        if len(self.dynamicComponents) > 0:
            self.dynamicEvaluation = self.dynamicComponents[0].evaluation
            for i in range(1, len(self.dynamicComponents)):
                comp = self.dynamicComponents[i]
                transition = mt.matrixAddInDiag(transition, comp.transition)
                self.dynamicEvaluation = mt.matrixAddByCol(self.dynamicEvaluation, \
                                                           comp.evaluation)
                state = mt.matrixAddByRow(state, comp.meanPrior)
                sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)

        # We then update the result in the base model
        evaluation = mt.matrixAddByCol(self.staticEvaluation, self.dynamicEvaluation)
        
        print 'Writing to the base model...'
        self.model = baseModel(transition = transition, \
                               evaluation = evaluation, \
                               noiseVar = noise, \
                               sysVar = sysVar, \
                               state = state, \
                               df = 0)
        self.model.initializeObservation()
        self.initialized = True
        print 'Initialization finished...'

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
        elif step >= self.dynamicComponents[0].n:
            raise NameError('The step is out of range')
        else:
            self.step = step

        # update the dynamic evaluation vector
        # We need first update all dynamic components by 1 step
        self.dynamicComponents[0].updateEvaluation(step)
        self.dynamicEvaluation = self.dynamicComponents[0].evaluation
        for i in range(1, len(self.dynamicEvaluation)):
            comp = self.dynamicComponents[i]
            comp.updateEvaluation(step)
            self.dynamicEvaluation = mt.matrixAddByCol(self.dynamicEvaluation, \
                                                       comp.evaluation)

        self.model.evaluation = mt.matrixAddByCol(self.staticEvaluation, \
                                                  self.dynamicEvaluation)
