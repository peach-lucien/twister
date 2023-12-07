""" The master template for an operation class.

Each operation class in the ./operations folder can inherit the main feature class functionality.

The functions here are necessary to evaluate each individual feature found inside a operation class.

"""
import logging
import multiprocessing
import warnings




warnings.simplefilter("ignore")
L = logging.getLogger(__name__)



class OperationClass:
    """ Main functionality to be inherited by each feature class"""

    # Class variables that describe the feature,
    # They should be defined for all child features


    def __init_subclass__(cls):
        """Initialise class variables to default for each child class."""


    def __init__(self, patient=None):
        """Initialise a feature class.

        Args:
            patient (Patient): patient for initialisation, converted to given encoding
        """
        self.pool = multiprocessing.Pool(processes=1)
        if patient is not None:
            self.patient = patient
            self.twister_predictions = patient.twister_predictions
            self.patient_id = patient.patient_id
        else:
            self.patient = None
            
        self.features = {}



    def get_features(self, all_features=False):
        """Compute all the possible features."""

        self.compute_features()

        return self.features




