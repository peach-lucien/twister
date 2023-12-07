import json
import importlib.resources
from pathlib import Path
from importlib import import_module
from functools import partial

from tqdm import tqdm

from twister.statistics.utils import NestedPool



def extract(patient_collection,
            stats_operations=None,
            n_workers=1,
            ):
    """Main function to perform statistics operations on the deep learning predictions."""  
    
    # get dictionary of statistics operations to perform
    if not stats_operations:
        stats_operations = get_stats_operations()
    
    # get the list of operation classes available
    feat_classes = get_list_operation_classes(stats_operations)    
    
    # perform all statistics operations
    stats = compute_all_features(
        patient_collection,
        feat_classes,
        n_workers=n_workers,
    )   
    
    # convert to dict
    stats = {k:v for element in stats for k,v in element.items()}
        
    return stats


def compute_all_features(
    patient_collection,
    list_feature_classes,
    n_workers=1,
):
    """ Compute all statistics operations """
    
    with NestedPool(n_workers) as pool:
        return list(
            tqdm(
                pool.imap(
                    partial(
                        feature_extraction,
                        list_feature_classes=list_feature_classes,
                    ),
                    patient_collection.patients,
                ),
                total=len(patient_collection),
            )
        )        
    
    
    
def feature_extraction(patient, list_feature_classes ):
    """Extract statistics for a single patient """
    
    features = {patient.patient_id: {}}
    for feature_class in list_feature_classes:

        # instantiate feature class object
        feat_class_inst = feature_class(patient)
        
        # extract features
        #features = pd.DataFrame(feat_class_inst.get_features(), index=[patient.patient_id])
        #features = {'somefeature',10}# feat_class_inst.get_features()
        
        features[patient.patient_id][type(feat_class_inst).__name__] = feat_class_inst.get_features()
        
        feat_class_inst.pool.close()
        
        # delete instance of class to save memory
        del feat_class_inst


    return features
    
    
    
    
    
def get_stats_operations():
    """ get dictionary of operations to perform """
    with importlib.resources.open_text("twister.statistics", "stats_operations.json") as file:
        stats_operations = json.load(file)  
    return stats_operations
    
def _load_feature_class(feature_name):
    """load the feature class from feature name."""
    feature_module = import_module("twister.statistics.operations." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)
   
    
def get_list_operation_classes(stats_operations):
    """ Get list of operations to perform on each patient """
    
    feature_path = Path(__file__).parent / "operations"
    non_feature_files = ["__init__", "utils"]

    list_feature_classes = []

    feature_names = [
        _f.stem for _f in feature_path.glob("*.py") if _f.stem not in non_feature_files
    ]   
    
    features = []
    for feature in feature_names:
        if stats_operations[feature] is True:
            features.append(feature)
            
   
    for feature_name in tqdm(features):
        feature_class = _load_feature_class(feature_name)        
        list_feature_classes.append(feature_class)           

    return list_feature_classes




















