"""
==============================================================================

Code for builder of a dynamic linear model

==============================================================================

This piece of code provides the basic functionality for constructing the model
of a dlm. It allows flexible modeling by users. User can add to, delete and
view componets of a given dlm. Builder will finally assemble all the components
 to a final big model.

"""
# this class provide all the model building operations for constructing
# customized model
import numpy as np
from pydlm.base.baseModel import baseModel
from copy import deepcopy
from pydlm.modeler.matrixTools import matrixTools as mt

# The builder will be the main class for construting dlm
# it featues two types of evaluation matrix and evaluation matrix
# The static evaluation remains the same over time which is used to
# record the trend and seasonality.
#
# The dynamic evaluation vector changes over time, it is basically
# the other variables that might have impact on the time series
# We need to update this vector as time going forward


class builder:
    """ The main modeling part of a dynamic linear model. It allows the users to
    custemize their own model. User can add, delete any components like trend
    or seasonality to the builder, or view the existing components. Builder
    will finally assemble all components to a big model for further training
    and inference.

    Attributes:
        model: the model structure from @baseModel, stores all the necessary
        quantities initialized: indicates whether the model has been built
        staticComponents: stores all the static components (trend and
                          seasonality)
        dynamicComponents: stores all the dynamic components
        componentIndex: the location of each component in the latent states
        statePrior: the prior mean of the latent state
        sysVarPrior: the prior of the covariance of the latent states
        noiseVar: the prior of the observation noise
        initialDegreeFreedom: the initial degree of freedom.
        discount: the discounting factor, please refer to @kalmanFilter for
                  more details
        renewTerm: used for aiding the stability of the model. The renewTerm
                   is computed according to the discount fact. When the filter
                   goes over certain steps, the information contribution of the
                   previous data has decayed to minimum. We then ignore those
                   days and refit the time series starting from current -
                   renewTerm. Thus, the effective sample size of the dlm is
                   twice renewTerm. When discount = 1, there will be no
                   renewTerm, since all the information will be passed along.
        renewDiscount: the minimum discount of seasonality component,
                       or if there is no seasonality, this will be the minimum
                       discount of all components. Used for computing
                       renewTerm.

    Methods:
        add: add new component
        ls:  list out all components
        delete: delete a specific component by its name
        initialize: assemble all the component to construt a big model
        updateEvaluation: update the valuation matrix of the big model
    """

    # create members
    def __init__(self):

        # the basic model structure for running kalman filter
        self.model = None
        self.initialized = False

        # to store all components. Separate the two as the evaluation
        # for dynamic component needs update each iteration
        # (new) added the third category, which is dynamic but the
        # features can be automatically computed from the data

        # store all the static components, i.e., the evaluation vector
        # do not change over time
        self.staticComponents = {}

        # store all the dynamic components, i.e., the evaluation vector
        # changes over time
        self.dynamicComponents = {}

        # store all the components that are dynamic, but the evaluation
        # vector can be inferred directly from the main data instead
        # of other sources
        self.automaticComponents = {}

        # store the index (location in the latent states) of all components
        # can be used to extract information for each componnet
        self.componentIndex = {}

        # record the prior guess on the latent state and system covariance
        self.statePrior = None
        self.sysVarPrior = None
        self.noiseVar = None
        self.initialDegreeFreedom = 1

        # record the discount factor for the model
        self.discount = None

        # renew term used to indicate the effective length of data, i.e.,
        # for days before this length will have little impact on
        # the current result
        self.renewTerm = -1.0
        self.renewDiscount = None  # used for adjusting renewTerm

        # flag for determining whether the system info should be printed.
        self._printInfo = True

    # The function that allows the user to add components
    def add(self, component):
        """ Add a new model component to the builder.

        Args:
            component: a model component, any class implements @component class

        """

        self.__add__(component)

    def __add__(self, component):
        if component.componentType == 'dynamic':
            if component.name in self.dynamicComponents:
                raise NameError('Please rename the component to a'
                                + ' different name.')
            self.dynamicComponents[component.name] = component

        if component.componentType == 'autoReg' \
           or component.componentType == 'longSeason':
            if component.name in self.automaticComponents:
                raise NameError('Please rename the component to a'
                                + ' different name.')
            self.automaticComponents[component.name] = component

        if component.componentType == 'trend' \
           or component.componentType == 'seasonality':
            if component.name in self.staticComponents:
                raise NameError('Please rename the component' +
                                ' to a different name.')
            self.staticComponents[component.name] = component

            # we use seasonality's discount to adjust the renewTerm
            if component.componentType == 'seasonality':
                if self.renewDiscount is None:
                    self.renewDiscount = 1.0
                self.renewDiscount = min(self.renewDiscount,
                                         min(component.discount))
        self.initialized = False
        return self

    # print all components to the client
    def ls(self):
        """ List out all the existing components to the model

        """

        if len(self.staticComponents) > 0:
            print('The static components are')
            for name in self.staticComponents:
                comp = self.staticComponents[name]
                print(comp.name + ' (degree = ' + str(comp.d) + ')')
            print(' ')
        else:
            print('There is no static component.')
            print(' ')

        if len(self.dynamicComponents) > 0:
            print('The dynamic components are')
            for name in self.dynamicComponents:
                comp = self.dynamicComponents[name]
                print(comp.name + ' (dimension = ' + str(comp.d) + ')')
            print(' ')
        else:
            print('There is no dynamic component.')
            print(' ')

        if len(self.automaticComponents) > 0:
            print('The automatic components are')
            for name in self.automaticComponents:
                comp = self.automaticComponents[name]
                print(comp.name + ' (dimension = ' + str(comp.d) + ')')
        else:
            print('There is no automatic component.')

    # delete the componet that pointed out by the client
    def delete(self, name):
        """ Delete a specific component from dlm by its name.

        Args:
            name: the name of the component. Can be read from ls()

        """

        if name in self.staticComponents:
            del self.staticComponents[name]
        elif name in self.dynamicComponents:
            del self.dynamicComponents[name]
        elif name in self.automaticComponents:
            del self.automaticComponents[name]
        else:
            raise NameError('Such component does not exisit!')

        self.initialized = False

    # initialize model for all the quantities
    # noise is the prior guess of the variance of the observed data
    # data is used by auto regressor.
    def initialize(self, data=[], noise=1):
        """ Initialize the model. It construct the baseModel by assembling all
        quantities from the components.

        Args:
            noise: the initial guess of the variance of the observation noise.
        """
        if len(self.staticComponents) == 0 and \
           len(self.dynamicComponents) == 0 and \
           len(self.automaticComponents) == 0:

            raise NameError('The model must contain at least' +
                            ' one component')

        # construct transition, evaluation, prior state, prior covariance
        if self._printInfo:
            print('Initializing models...')
        transition = None
        evaluation = None
        state = None
        sysVar = None
        self.discount = np.array([])

        # first construct for the static components
        # the evaluation will be treated separately for static or dynamic
        # as the latter one will change over time
        currentIndex = 0  # used for compute the index
        for i in self.staticComponents:
            comp = self.staticComponents[i]
            transition = mt.matrixAddInDiag(transition, comp.transition)
            evaluation = mt.matrixAddByCol(evaluation,
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
                comp.updateEvaluation(0)
                transition = mt.matrixAddInDiag(transition, comp.transition)
                evaluation = mt.matrixAddByCol(evaluation,
                                               comp.evaluation)
                state = mt.matrixAddByRow(state, comp.meanPrior)
                sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)
                self.discount = np.concatenate((self.discount, comp.discount))
                self.componentIndex[i] = (currentIndex,
                                          currentIndex + comp.d - 1)
                currentIndex += comp.d

        # if the model contains the automatic dynamic part, we add
        # them to the builder
        if len(self.automaticComponents) > 0:
            self.automaticEvaluation = None
            for i in self.automaticComponents:
                comp = self.automaticComponents[i]
                comp.updateEvaluation(0, data)
                transition = mt.matrixAddInDiag(transition, comp.transition)
                evaluation = mt.matrixAddByCol(evaluation,
                                               comp.evaluation)
                state = mt.matrixAddByRow(state, comp.meanPrior)
                sysVar = mt.matrixAddInDiag(sysVar, comp.covPrior)
                self.discount = np.concatenate((self.discount, comp.discount))
                self.componentIndex[i] = (currentIndex,
                                          currentIndex + comp.d - 1)
                currentIndex += comp.d

        self.statePrior = state
        self.sysVarPrior = sysVar
        self.noiseVar = np.matrix(noise)
        self.model = baseModel(transition=transition,
                               evaluation=evaluation,
                               noiseVar=np.matrix(noise),
                               sysVar=sysVar,
                               state=state,
                               df=self.initialDegreeFreedom)
        self.model.initializeObservation()

        # compute the renew period
        if self.renewDiscount is None:
            self.renewDiscount = np.min(self.discount)

        if self.renewDiscount < 1.0 - 1e-8:
            self.renewTerm = np.log(0.001 * (1 - self.renewDiscount)) \
                             / np.log(self.renewDiscount)

        self.initialized = True
        if self._printInfo:
            print('Initialization finished.')

    # Initialize from another builder exported from other dlm class
    def initializeFromBuilder(self, data, exported_builder):
        # Copy the components
        self.staticComponents = deepcopy(exported_builder.staticComponents)
        self.automaticComponents = deepcopy(exported_builder.automaticComponents)
        self.dynamicComponents = deepcopy(exported_builder.dynamicComponents)
        self.componentIndex = deepcopy(exported_builder.componentIndex)
        self.discount = deepcopy(exported_builder.discount)
        self.initialDegreeFreedom = exported_builder.model.df

        # Copy the model states
        self.statePrior = exported_builder.model.state
        self.sysVarPrior = exported_builder.model.sysVar
        self.noiseVar = exported_builder.model.noiseVar
        self.transition = exported_builder.transition
        self.evaluation = exported_builder.evaluation
        # update the evaluation to the current.
        self.updateEvaluation(step=0, data=data)

    
        self.model = baseModel(transition=self.transition,
                               evaluation=self.evaluation,
                               noiseVar=self.noiseVar,
                               sysVar=self.sysVarPrior,
                               state=self.statePrior,
                               df=self.initialDegreeFreedom)
        self.model.initializeObservation()

        # compute the renew period
        if self.renewDiscount is None:
            self.renewDiscount = np.min(self.discount)

        if self.renewDiscount < 1.0 - 1e-8:
            self.renewTerm = np.log(0.001 * (1 - self.renewDiscount)) \
                             / np.log(self.renewDiscount)
        
        self.initialized = True
        if self._printInfo:
            print('Initialization finished.')
            
    # This function allows the model to update the dynamic evaluation vector,
    # so that the model can handle control variables
    # This function should be called only when dynamicComponents is not empty
    # data is used by auto regressor.
    def updateEvaluation(self, step, data):
        """ Update the evaluation matrix of the model to a specific date.
        It loops over all dynamic components and update their evaluation
        matrix and then reconstruct the model evaluation matrix by
        incorporating the new evaluations

        Arges:
            step: the date at which the evaluation matrix is needed.

        """

        # if len(self.dynamicComponents) == 0 and \
        #   len(self.automaticComponents) == 0:
        #    raise NameError('This shall only be used when there' +
        #                    ' are dynamic or automatic components!')

        # update the dynamic evaluation vector
        # We need first update all dynamic components by 1 step
        for i in self.dynamicComponents:
            comp = self.dynamicComponents[i]
            comp.updateEvaluation(step)
            self.model.evaluation[0, self.componentIndex[i][0]:
                                  (self.componentIndex[i][1] + 1)] = comp.evaluation

        for i in self.automaticComponents:
            comp = self.automaticComponents[i]
            comp.updateEvaluation(step, data)
            self.model.evaluation[0, self.componentIndex[i][0]:
                                  (self.componentIndex[i][1] + 1)] = comp.evaluation
