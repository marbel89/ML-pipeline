import pandas as pd
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
from dataSet import DataSet

# Creating a DataSet object for customer churn data
# - feature_list: List of features to be used in the model
# - file_name: Path to the CSV file containing the data
# - label_col: The name of the column that contains the labels
# - pos_category: The category value in the label column representing a positive case

customer_obj = DataSet(
    feature_list=["total_day_minutes",
                  "total_day_calls",
                  "number_customer_service_calls"],
    file_name="../data/customer_churn_data.csv",
    label_col="churn",
    pos_category="yes"
)

"""
# Gradient Boosting is an ensemble learning technique that builds the model in a stage-wise fashion.
# It is typically used for classification and regression tasks, known for its accuracy and robustness.
# The GBM model combines the predictions of several base estimators (weak learners) to improve overall performance.

# Creating a Gradient Boosting Classifier model with specified hyperparameters:
# - learning_rate: The step size shrinkage used in each boosting step, controlling overfitting (0.1 in this case).
# - n_estimators: The number of boosting stages to be run, more estimators usually improve performance (here: 300)
# - subsample: The fraction of samples to be used for fitting the individual base learners (0.7 in this case).
# - min_samples_split: The minimum number of samples required to split an internal node, controlling overfitting (40).
# - max_depth: The maximum depth of individual regression estimators, controlling the complexity of each tree (here: 3).
"""

gbm_model = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.7,
    min_samples_split=40,
    max_depth=3)


# Trains the model with the features and labels from customer_obj
gbm_model.fit(customer_obj.train_features, customer_obj.train_labels)

# Creates a dataframe and adds the customer ids
output = pd.DataFrame([index for index in range(customer_obj.test_features.shape[0])], columns=["customer_id"])

# Applies the prediction on the test dataset. [::1] takes the entire array
output["model_prediction"] = gbm_model.predict_proba(customer_obj.test_features)[::1]

# todo
output["prediction_date"] = "2023-04-01"

con = sqlite3.connect("data/customers.db")
cursor = con.cursor()

insert_command = """
INSERT INTO
customer_churn_predictions(
customer_id,
model_prediction,
prediction_date)
values(?,?,?) """

cursor.executemany(insert_command, output.values)

con.commit()

query = """SELECT * FROM customer_churn_predictions"""
cursor.execute(query)
customer_records = cursor.fetchall()
cursor.close()

### NExt steps: Classing Ml train und config
