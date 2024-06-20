import sys
from sklearn.ensemble import GradientBoostingClassifier
from dataSet import DataSet
from ml_model import MLModel

import json
import joblib
import argparse

# Adding the current directory to the system path to ensure modules can be imported correctly
sys.path.append(".")

# Setting up argument parser to handle configuration file input
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args()

# Loading configuration from the provided JSON file
with open(args.config, "r") as config_file:
    config = json.load(config_file)

# Creating a DataSet object using configuration parameters
customer_obj = DataSet(
    feature_list=config["input_data"]["needed_fields"],
    file_name=config["input_data"]["filename"],
    label_col=config["input_data"]["label_col"],
    pos_category=config["input_data"]["pos_category"]
)

# Extracting model parameters from the configuration
model_parameters = config["model_parameters"]

# Adjusting ranges for specific model parameters
model_parameters["min_samples_leaf"] = range(model_parameters["min_samples_leaf_lower"],
                                             model_parameters["min_samples_leaf_upper"],
                                             model_parameters["min_samples_leaf_inc"])

model_parameters["min_samples_split"] = range(model_parameters["min_samples_split_lower"],
                                              model_parameters["min_samples_split_upper"],
                                              model_parameters["min_samples_split_inc"])

# Defining a set of parameters to be checked and filtered
parameter_check = {"max_depth", "subsample", "max_features", "n_estimators", "learning_rate",
                   "min_samples_leaf", "min_samples_split"}

# Filtering model parameters to include only those specified in parameter_check
model_parameters = {key: val for key, val in model_parameters.items() if key in parameter_check}


# Creating an instance of MlModel with the GradientBoostingClassifier and tuning parameters
gbm = MLModel(
    ml_model=GradientBoostingClassifier(),
    parameters=model_parameters,
    n_jobs=config["hyperparameter_settings"]["n_jobs"],
    scoring=config["hyperparameter_settings"]["scoring"],
    n_iter=config["hyperparameter_settings"]["n_iter"],
    random_state=0
)

# Tuning the model using the training data
gbm.tune(customer_obj.train_features, customer_obj.train_labels)

# Saving the trained model to a file
joblib.dump(gbm, config["model_filename"])
print("Model trained and saved to ", config["model_filename"])
