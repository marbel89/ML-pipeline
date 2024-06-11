import sys
from sklearn.ensemble import GradientBoostingClassifier
from dataSet import DataSet
from ml_model import MLModel
import json
import joblib
import argparse
sys.path.append(".")

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config")

args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)

customer_obj = DataSet(
    feature_list=config["input_data"]["needed_fields"],
    file_name=config["input_data"]["filename"],
    label_col=config["input_data"]["label_col"],
    pos_category=config["input_data"]["pos_category"]
)

model_parameters = config["model_parameters"]

model_parameters["min_samples_leaf"] = range(model_parameters["min_samples_leaf_lower"],
                                             model_parameters["min_samples_leaf_upper"],
                                             model_parameters["min_samples_leaf_inc"])

model_parameters["min_samples_split"] = range(model_parameters["min_samples_split_lower"],
                                              model_parameters["min_samples_split_upper"],
                                              model_parameters["min_samples_split_inc"])

parameter_check = {"max_depth",
                   "subsample", "max_features",
                   "n_estimators", "learning_rate",
                   "min_samples_leaf", "min_samples_split"}

model_parameters = {key: val for key, val in
                    model_parameters.items() if
                    key in parameter_check}



gbm = MlModel(
    ml_model=GradientBoostingClassifier(),
    parameters=model_parameters,
    n_jobs=config["hyperparameter_settings"]["n_jobs"],
    scoring=config["hyperparameter_settings"]["scoring"],
    n_iter=config["hyperparameter_settings"]["n_iter"],
    random_state=0)

gbm.tune(customer_obj.train_features,
         customer_obj.train_labels)

joblib.dump(gbm, config["model_filename"])
print("Model trained and saved to ",
      config["model_filename"])
