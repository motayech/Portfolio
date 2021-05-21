# Used Car Price Prediction

This project was a part of my machine learning course in my masters program. The notebook has five main sections:
* I - Baseline System: Some data-preprocessing is done on the training set and a linear regression model is built on the training set. The model is evaluated using 5-fold cross validation.
* II - Feature Engineering: To lower the RMSE of our baseline model, in-depth feature engineering is conducted. The RMSE was successfully decreased after feature engineering.
* III - Model Optimization & Selection: The hyperparameters of three models (Decision Tree, Random Forest, Gradient Boosting) was optimized using GridSearchCV. Once the optimal hyperparameters are determined, the models were built and evaluated. All models were successful in lowering the RMSE.
* IV - Feature Selection: Unimportant features were determined and dropped from our optimal model. The RMSE was further lowered after dropping these features.
* V - Pipeline, Interpretable Model, and Final Testing: A pipeline was built to automate the pre-processing on both the training and the test set. However, many pre-processing stages are not implemented in the Pipeline due to the creation of new features in Section II - Feature Engineering. In addition, further hyperparameter tuning was done for an interpretable model. Finally, the models were all evaluated on the test set to determine the optimal model to predict used car prices.
