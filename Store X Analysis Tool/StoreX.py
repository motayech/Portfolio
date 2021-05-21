import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = 'Store X Analysis Tool', layout = 'wide')

dataset = st.file_uploader("Please upload your dataset of interest here!", type = ["csv"])

if dataset == None:
    st.title("Store X Analysis Tool")
    st.write("Welcome to the Store X Analysis Tool! This tool enables us to view our dataset, gain insights on customers, sales, purchases, and campaigns with the use of dashboards, and even predict whether a customer is likely to respond to a campaign or not!")
    st.write("In all businesses, it is vital every once in a while to carry out promotional campaigns for your products or services; these campaigns are likely to increase the sales of the business. However, in order to increase sales at an optimal level, these promotions must be sent to customers who are likely to utilize and respond to these campaigns. As a result, the indentifiication of these customers is significant and vital in the success of the campaign.")
    st.subheader("To begin identifying the customers who are most likely to accept our offers, simply upload the dataset above!")
    st.image('https://cdn-a.william-reed.com/var/wrbm_gb_food_pharma/storage/images/publications/food-beverage-nutrition/foodnavigator-asia.com/headlines/markets/can-unmanned-convenience-stores-take-off-in-indonesia-jd.com-thinks-so/8506049-1-eng-GB/Can-unmanned-convenience-stores-take-off-in-Indonesia-JD.com-thinks-so.jpg')
else:
    st.title("Store X Analysis Tool")
    st.image('https://previews.123rf.com/images/vectorpouch/vectorpouch1809/vectorpouch180900131/110754541-supermarket-and-grocery-food-products-on-shelves-vector-illustration-no-people-on-cartoon-background.jpg')
    data = pd.read_csv(dataset)
    data = data.dropna()
    data = data[data['Year_Birth'] >= 1940]
    data = data[data['Income'] < 200000]

    data_original = data.copy()
    data_original = data_original.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'MntGoldProds', 'Kidhome', 'Teenhome', 'NumCatalogPurchases'])

    couple = ['Married', 'Together']
    c_list = []

    for x in list(data_original['Marital_Status']):
        if x in couple:
            c_list.append(1)
        else:
            c_list.append(0)

    c_list = pd.Series(c_list)
    data_original['Is_Coupled'] = c_list.values
    data_original = data_original.drop(columns = ['Marital_Status'], axis = 1)

    X = data.drop(columns = ['Response'], axis = 1)
    y = data['Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_train = X_train.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'MntGoldProds', 'Kidhome', 'Teenhome', 'NumCatalogPurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'])

    X_train['Total_Spent'] = X_train['MntWines'] + X_train['MntFruits'] + X_train['MntMeatProducts'] + X_train['MntFishProducts'] + X_train['MntSweetProducts']
    X_train['Total_Purchases'] = X_train['NumWebPurchases'] + X_train['NumStorePurchases']

    X_train = X_train.drop(columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'NumWebPurchases', 'NumStorePurchases'])

    X_train["Education"] = X_train["Education"].astype('category')
    X_train["Education"] = X_train["Education"].cat.codes

    couple = ['Married', 'Together']
    c_list = []

    for x in list(X_train['Marital_Status']):
        if x in couple:
            c_list.append(1)
        else:
            c_list.append(0)

    c_list = pd.Series(c_list)
    X_train['Is_Coupled'] = c_list.values
    X_train = X_train.drop(columns = ['Marital_Status'], axis = 1)

    X_test = X_test.drop(columns = ['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'MntGoldProds', 'Kidhome', 'Teenhome', 'NumCatalogPurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'])

    X_test['Total_Spent'] = X_test['MntWines'] + X_test['MntFruits'] + X_test['MntMeatProducts'] + X_test['MntFishProducts'] + X_test['MntSweetProducts']
    X_test['Total_Purchases'] = X_test['NumWebPurchases'] + X_test['NumStorePurchases']
    X_test = X_test.drop(columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'NumWebPurchases', 'NumStorePurchases'])

    X_test["Education"] = X_test["Education"].astype('category')
    X_test["Education"] = X_test["Education"].cat.codes

    couple = ['Married', 'Together']
    c_list = []

    for x in list(X_test['Marital_Status']):
        if x in couple:
            c_list.append(1)
        else:
            c_list.append(0)

    c_list = pd.Series(c_list)
    X_test['Is_Coupled'] = c_list.values
    X_test = X_test.drop(columns = ['Marital_Status'], axis = 1)


if dataset != None:

    page = st.selectbox("Select the page of interest", ["Data Overview", "Dashboards",  "Prediction Models", "Predicting Response to Campaigns"])

    if page == "Data Overview":

        if st.checkbox("Click here to view the dataset!"):
            n = st.slider("How many rows of the dataset do you want to see?", min_value = 1, max_value = data_original.shape[0])
            st.dataframe(data_original.head(n))
    
    if page == 'Dashboards':
        dashboard_choice = st.radio('What dashboard would you like to see?', ('Customer Overview', 'Sales Breakdown', 'Purchase Breakdown & Complaints', 'Previous Campaign Performance'))

        if dashboard_choice == 'Customer Overview':
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("**Customer Year of Birth**")
                fig = px.histogram(data_original, x = 'Year_Birth', nbins = 12, labels = {'Year_Birth': 'Birth Year'}, title = 'Histogram of Year of Birth', width = 550, height = 500, color_discrete_sequence = ['#10A898'])
                st.plotly_chart(fig)

                if st.checkbox('Show analysis of customer year of birth'):
                    st.write("The histogram of customers' year of birth indicates that the majority of our customers are born in the year " + str(data_original['Year_Birth'].mode()[0]) + '. Our oldest customer was born in the year ' + str(data_original['Year_Birth'].min()) + ', while our youngest customer was born in the year ' + str(data_original['Year_Birth'].max()) + '.')
                    st.write("As a result, most of our customers are " + str(date.today().year - data_original['Year_Birth'].mode()[0]) + " years old, our oldest customer is " + str(date.today().year - data_original['Year_Birth'].min()) + ' years old, and our youngest customer is ' + str(date.today().year - data_original['Year_Birth'].max()) + ' years old.')
            
            with col2:
                st.subheader("**Customer Education Level**")
                fig = px.histogram(data_original, x = 'Education', color = 'Education', labels = {'Education': 'Education Level'}, title = "Bar Plot of Customer Education Levels", width = 550, height = 500, color_discrete_map = {'Graduation': '#3FBB94', 'PhD': '#69CE8C', 'Master': '#96DE81', 'Basic': '#C5EC77', '2n Cycle': '#F9F871'})
                st.plotly_chart(fig)

                if st.checkbox('Show analysis of customer education'):
                    index = data_original['Education'].value_counts().index.tolist()
                    values = data_original['Education'].value_counts().values.tolist()
                    st.write('As we can see from the bar chart, the majority of our customers have a ' + str(index[0]) + ' education, with ' + str(values[0]) + ' customers possessing this level of education. Only a few customers have a ' + str(index[4]) + ' education.')
            
            st.markdown("<hr/>", unsafe_allow_html = True)

            st.subheader("**Customer Relationship Status**")
            statement = 'Percentage of customers who are not in a relationship 🔓'
            st.markdown(f"<h1 style ='text-align: center; color: black;'>{statement}</h1>", unsafe_allow_html = True)
            single = ((data_original['Is_Coupled'].value_counts()[0] / data_original['Is_Coupled'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%' 
            st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{single}</h1>", unsafe_allow_html = True)

            statement = 'Percentage of couples who are in a relationship 💑'
            st.markdown(f"<h1 style ='text-align: center; color: black;'>{statement}</h1>", unsafe_allow_html = True)
            couple = ((data_original['Is_Coupled'].value_counts()[1] / data_original['Is_Coupled'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
            st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{couple}</h1>", unsafe_allow_html = True)

            st.markdown("<hr/>", unsafe_allow_html = True)

            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("**Customer Income**")
                fig = px.histogram(data_original, x = 'Income', nbins = 20, title = 'Histogram of Customer Income Levels', width = 550, height = 500, color_discrete_sequence = ['#10A898'])
                st.plotly_chart(fig)

                if st.checkbox('Show analysis of customer income'):
                    st.write("The histogram above seems to indicate that the median yearly household income of our customers is roughly equal to " + str(data_original['Income'].median()) + '. Our lowest income customer has an income of ' + str(data_original['Income'].min()) + ', while our highest income customer receives an income of ' + str(data_original['Income'].max()) + '.')
            
            with col2:
                st.subheader('**Customer Recency**')
                fig = px.histogram(data_original, x = 'Recency', nbins = 12, labels = {'Recency': 'Recency (in days)'}, title = 'Histogram of Customer Income Levels', width = 550, height = 500, color_discrete_sequence = ['#96B1AC'])
                st.plotly_chart(fig)

                if st.checkbox('Show analysis of customer recency'):
                    st.write('On average, the number of days since the last purchase of customers is equal to ' + str(int(data_original['Recency'].mean())) + ' days. The longest number of days since the last purchase of a customer is ' + str(int(data_original['Recency'].max())) + ' days, while the shortest number of days since the last purchase of a customer is equal to ' + str(int(data_original['Recency'].min())) + ' days.')


        if dashboard_choice == 'Sales Breakdown':
            st.subheader("**Our Products**")

            wine, fruit, meat, fish, sweets = st.beta_columns(5)

            with wine:
                st.markdown('**Wines**')
                st.image('https://cdn2.iconfinder.com/data/icons/pittogrammi/142/98-512.png', width = 125)
        
            with fruit:
                st.markdown('**Fruit**')
                st.image('https://cdn0.iconfinder.com/data/icons/food-set-4/64/Artboard_12_copy-512.png', width = 125)

            with meat:
                st.markdown('**Meat**')
                st.image('https://cdn0.iconfinder.com/data/icons/food-set-4/64/Artboard_6-512.png', width = 125)
        
            with fish:
                st.markdown('**Fish**')
                st.image('https://icon-library.com/images/fish-food-icon/fish-food-icon-0.jpg', width = 125)
        
            with sweets:
                st.markdown('**Sweets**')
                st.image('https://www.shareicon.net/data/512x512/2016/03/22/737942_food_512x512.png', width = 125)
        
            st.markdown("<hr/>", unsafe_allow_html = True)

            wine_sales = data_original['MntWines'].sum()
            fruit_sales = data_original['MntFruits'].sum()
            meat_sales = data_original['MntMeatProducts'].sum()
            fish_sales = data_original['MntFishProducts'].sum()
            sweet_sales = data_original['MntSweetProducts'].sum()

            total_sales = wine_sales + fruit_sales + meat_sales + fish_sales + sweet_sales

            st.subheader('**Total Sales**')
            sales_value = str('${:,.2f}'.format(total_sales))
            st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{sales_value}</h1>", unsafe_allow_html = True)

            st.markdown("<hr/>", unsafe_allow_html = True)

            st.subheader('**Distribution of Sales**')
            sales_dict = {'Category': ['Wines', 'Fruit', 'Meat', 'Fish', 'Sweets'], 'Sales': [wine_sales, fruit_sales, meat_sales, fish_sales, sweet_sales]}
            sales_data = pd.DataFrame(sales_dict, columns = ['Category', 'Sales'])

            fig = px.pie(sales_data, names = 'Category', values = 'Sales', color = 'Category', width = 1125, height = 500, title = 'Distribution of Sales: Pie Chart', color_discrete_map = {'Wines': '#3FBB94', 'Fruit': '#69CE8C', 'Meat': '#96DE81', 'Fish': '#C5EC77', 'Sweets': '#F9F871'})
            st.plotly_chart(fig)

            if st.checkbox("Click here to see the sales amounts of each category"):
                st.write("The total sales for wines is equal to " + str('${:,.2f}'.format(wine_sales)))
                st.write("The total sales for fruits is equal to " + str('${:,.2f}'.format(fruit_sales)))
                st.write("The total sales for meat is equal to " + str('${:,.2f}'.format(meat_sales)))
                st.write("The total sales for fish is equal to " + str('${:,.2f}'.format(fish_sales)))
                st.write("The total sales for sweets is equal to " + str('${:,.2f}'.format(sweet_sales)))
   
        if dashboard_choice == 'Purchase Breakdown & Complaints':
            st.subheader('**Our Stores**')
            online, store = st.beta_columns(2)

            with online:
                st.markdown('**Online Store**')
                st.image('https://www.shareicon.net/data/512x512/2016/04/16/750682_laptop_512x512.png', width = 230)
        
            with store:
                st.markdown('**Physical Store**')
                st.image('https://i.pinimg.com/originals/77/e5/0c/77e50c04f9f512a456eb3e135a1c013b.png', width = 230)
        
            st.markdown("<hr/>", unsafe_allow_html = True)
            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("**Total Units Purchased**")

                online_purchases = data_original['NumWebPurchases'].sum()
                store_purchases = data_original['NumStorePurchases'].sum()
                total_purchases = online_purchases + store_purchases

                purchases_value = str('{:,}'.format(total_purchases)) + ' units'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{purchases_value}</h1>", unsafe_allow_html = True)
            
            with col2:
                st.subheader("**Percentage of Customers Visiting Online Store and Carrying out a Purchase**")
                online_store_percentage = (data_original['NumWebPurchases'].sum() / data_original['NumWebVisitsMonth'].sum() * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{online_store_percentage}</h1>", unsafe_allow_html = True)

            st.markdown("<hr/>", unsafe_allow_html = True)

            st.subheader("**Store Purchase Distribution**")
            purchases_dict = {'Purchase Location': ['Online', 'Physical'], 'Purchases': [online_purchases, store_purchases]}
            purchases_data = pd.DataFrame(purchases_dict, columns = ['Purchase Location', 'Purchases'])
            
            fig = px.pie(purchases_data, names = 'Purchase Location', values = 'Purchases', color = 'Purchase Location', width = 1125, height = 500, title = 'Distribution of Purchases: Pie Chart', color_discrete_map = {'Online': '#3FBB94', 'Physical': '#96DE81'})
            st.plotly_chart(fig)

            if st.checkbox("Click here to see the units purchased in each store"):
                st.write(str('{:,}'.format(online_purchases)) + ' units have been purchased from the online store')
                st.write(str('{:,}'.format(store_purchases)) + ' units have been purchased from the physical store')
        
            st.markdown("<hr/>", unsafe_allow_html = True)

            col1, col2 = st.beta_columns(2)

            with col1:
                st.subheader("**Percentage of Discounted Purchases**")
                discount_percentage = (data_original['NumDealsPurchases'].sum() / total_purchases * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{discount_percentage}</h1>", unsafe_allow_html = True)

            with col2:
                st.subheader("**Percentage of Customers Issuing a Complaint**")
                complaint_percentage = (data_original['Complain'].value_counts()[1] / data_original['Complain'].value_counts().sum() * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{complaint_percentage}</h1>", unsafe_allow_html = True)

        if dashboard_choice == 'Previous Campaign Performance':
            st.subheader("**Success Rate of Previous Campaigns**")
            cm1, cm2, cm3 = st.beta_columns(3)

            with cm1:
                st.markdown('**Campaign 1**')
                number1 = ((data_original['AcceptedCmp1'].value_counts()[1] / data_original['AcceptedCmp1'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{number1}</h1>", unsafe_allow_html = True)

            with cm2:
                st.markdown('**Campaign 2**')
                number2 = ((data_original['AcceptedCmp2'].value_counts()[1] / data_original['AcceptedCmp2'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{number2}</h1>", unsafe_allow_html = True)

            with cm3:
                st.markdown('**Campaign 3**')
                number3 = ((data_original['AcceptedCmp3'].value_counts()[1] / data_original['AcceptedCmp3'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{number3}</h1>", unsafe_allow_html = True)

            st.markdown("<hr/>", unsafe_allow_html = True)

            cm4, cm5 = st.beta_columns(2)

            with cm4:
                st.markdown('**Campaign 4**')
                number4 = ((data_original['AcceptedCmp4'].value_counts()[1] / data_original['AcceptedCmp4'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{number4}</h1>", unsafe_allow_html = True)

            with cm5:
                st.markdown('**Campaign 5**')
                number5 = ((data_original['AcceptedCmp5'].value_counts()[1] / data_original['AcceptedCmp5'].value_counts().sum()) * 100).round(decimals = 2).astype('str') + '%'
                st.markdown(f"<h1 style ='text-align: center; color: #344B47;'>{number5}</h1>", unsafe_allow_html = True)
        
            st.markdown("<hr/>", unsafe_allow_html = True)

            st.subheader('**Customer Response to Campaigns**')
            cust_accept_offer = [data_original['AcceptedCmp1'].value_counts()[1], data_original['AcceptedCmp2'].value_counts()[1], 
            data_original['AcceptedCmp3'].value_counts()[1], data_original['AcceptedCmp4'].value_counts()[1], 
            data_original['AcceptedCmp5'].value_counts()[1]]

            campaign_number = [1,2,3,4,5]

            campaign_dict = {'Campaign Number': campaign_number, 'Number of Customers who Accepted Offer': cust_accept_offer}
            cust_accept_data = pd.DataFrame(campaign_dict, columns = ['Campaign Number', 'Number of Customers who Accepted Offer'])

            st.table(cust_accept_data)
            fig = px.line(cust_accept_data, x = 'Campaign Number', y = 'Number of Customers who Accepted Offer', title = 'Number of Customers Responding to Campaigns', width = 1175, color_discrete_sequence = ['#344B47'])
            st.plotly_chart(fig)
    
    if page == 'Prediction Models':
        alg = st.radio("What supervised ML algorithm would you like to see in predicting whether a customer will respond to a campaign or not?", ('Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest'))

        if alg == 'Logistic Regression':
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            acc = accuracy_score(y_pred, y_test)

            st.write("Logistic regression accurately predicts whether a customer will respond to a campaign or not " + str("{:.2%}".format(acc)) + " of the time")

            cf, roc = st.beta_columns(2)

            with cf:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(lr, X_test, y_test)
                st.pyplot()

            with roc:
                st.subheader("ROC Curve")
                plot_roc_curve(lr, X_test, y_test)
                st.pyplot()
            
        if alg == 'KNN':
            knn = KNeighborsClassifier(n_neighbors = 10)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_pred, y_test)

            st.write("KNN accurately predicts whether a customer will respond to a campaign or not " + str("{:.2%}".format(acc)) + " of the time")

            cf, roc = st.beta_columns(2)

            with cf:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(knn, X_test, y_test)
                st.pyplot()

            with roc:
                st.subheader("ROC Curve")
                plot_roc_curve(knn, X_test, y_test)
                st.pyplot()
        
        if alg == 'Decision Tree':
            dt = DecisionTreeClassifier(max_depth = 6, max_features = 0.75, random_state = 42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            acc = accuracy_score(y_pred, y_test)

            st.write("Decision trees accurately predicts whether a customer will respond to a campaign or not " + str("{:.2%}".format(acc)) + " of the time")

            cf, roc = st.beta_columns(2)

            with cf:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(dt, X_test, y_test)
                st.pyplot()

            with roc:
                st.subheader("ROC Curve")
                plot_roc_curve(dt, X_test, y_test)
                st.pyplot()
        
        if alg == 'Random Forest':
            rf = RandomForestClassifier(max_depth = 6, max_features = 0.75, random_state = 42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_pred, y_test)

            st.write("Random forests accurately predicts whether a customer will respond to a campaign or not " + str("{:.2%}".format(acc)) + " of the time")

            cf, roc = st.beta_columns(2)

            with cf:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(rf, X_test, y_test)
                st.pyplot()

            with roc:
                st.subheader("ROC Curve")
                plot_roc_curve(rf, X_test, y_test)
                st.pyplot()

    if page == 'Predicting Response to Campaigns':
        rf = RandomForestClassifier(max_depth = 6, max_features = 0.75, random_state = 42)
        rf.fit(X_train, y_train)

        def predict_outcome(Year_Birth, Education, Income, Is_Coupled, Recency, Total_Spent, NumDealsPurchases, Total_Purchases, NumWebVisitsMonth, Complain):

            if Education == '2n Cycle':
                Education = 0
            elif Education == 'Basic':
                Education = 1
            elif Education == 'Graduation':
                Education = 2
            elif Education == 'Master':
                Education = 3
            else:
                Education = 4
            
            if Is_Coupled == 'Yes':
                Is_Coupled = 1
            else:
                Is_Coupled = 0
            
            if Complain == 'Yes':
                Complain = 1
            else:
                Complain = 0
            

            pred_model = rf.predict([[Year_Birth, Education, Income, Is_Coupled, Recency, Total_Spent, NumDealsPurchases, Total_Purchases, NumWebVisitsMonth, Complain]])

            if pred_model == 0:
                pred = 'not respond to the campaign'
            if pred_model == 1:
                pred = 'respond to the campaign'
            
            return pred
        
        Year_Birth = st.number_input("What year was the customer born?", min_value = 1940, step = 1)
        Education = st.selectbox("What is the customer's highest level of education?", ('2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'))
        Income = st.number_input("What is the customer's income?")
        Is_Coupled = st.selectbox("Is the customer in a relationship?", ('Yes', 'No'))
        Recency = st.number_input("How many days has it been since the customer visited the store?", min_value = 0, step = 1)
        Total_Spent = st.number_input("How much has the customer spent in total across all product categories?")
        NumDealsPurchases = st.number_input("How many discounted purchases has the customer made?", min_value = 0, step = 1)
        Total_Purchases = st.number_input("How many products has the customer purchased from our stores?", min_value = 0, step = 1)
        NumWebVisitsMonth = st.number_input("How many times has the customer visited the store website?", min_value = 0, step = 1)
        Complain = st.selectbox("Has the customer ever issued a complaint?", ('Yes', 'No'))

        if st.button("Predict"):
            result = predict_outcome(Year_Birth, Education, Income, Is_Coupled, Recency, Total_Spent, NumDealsPurchases, Total_Purchases, NumWebVisitsMonth, Complain)
            st.success("The customer is likely to {}".format(result))


################################################################################################################################################

# REFERENCES:
# https://www.youtube.com/watch?v=tx6bT2Sh9R8
# https://www.kaggle.com/rodsaldanha/arketing-campaign
# https://www.analyticsvidhya.com/blog/2020/12/deploying-machine-learning-models-using-streamlit-an-introductory-guide-to-model-deployment/

################################################################################################################################################
