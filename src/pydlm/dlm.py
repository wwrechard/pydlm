# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

#from pydlm.modeler.builder import builder
from pydlm.func.dlm_func import dlm_func


class dlm(dlm_func):

    # define the basic members
    # initialize the result
    def __init__(self, data):
        dlm_func.__init__(data)

    # add component
    def add(self, component):
        self.builder.add(component)

    # list all components
    def ls(self):
        self.builder.ls()

    # delete one component
    def delete(self, index):
        self.builder.delete(index)

    
