# Store X Analysis Tool

The tool was built for a digital marketing course in my masters program. The link to the tool is: https://share.streamlit.io/motayech/portfolio/main/Store-X-Analysis-Tool/StoreX.py

Once the dataset (in the folder) is uploaded, the user has a choice of four pages to view:
* **Data Overview**: Gives a simple overview of the uploaded dataset. The user can select how many rows to view using a slider.
* **Dashboards**: Four main dashboard are constructed: a customer overview dashboard, a sales breakdown dashboard, a purchase breakdown & complaints dashboard, and a previous campaign performance dashboard. The user can select which dashboard to view using a radio button. In addition, within some dashboards there are some checkboxes to view additional analysis.
* **Prediction Models**: Four supervised ML models are built to predict whether a customer will respond to a promotional campaign or not. The selection of the ML model is done by the user using a radio button. The confusion matrix and ROC curve are plotted for each model.
* **Predicting Response to Campaigns**: A random forest model is used to predict customer response to campaigns. Once certain customer information is entered, the user can simply generate a prediction using the "Predict" button.

The prediction tool was built with the help of the following article: https://www.analyticsvidhya.com/blog/2020/12/deploying-machine-learning-models-using-streamlit-an-introductory-guide-to-model-deployment/

The dataset was retrieved from Kaggle: https://www.kaggle.com/rodsaldanha/arketing-campaign
