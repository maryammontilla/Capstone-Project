import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Set page title and icon
st.set_page_config(page_title="Hair Loss Prediction", page_icon=":bald_man:")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Modeling", "Make Predictions!"])

df = pd.read_csv('data/df1.csv')
df_eda = pd.read_csv('data/df-eda.csv')
# Home Page
if page == "Home":
    st.title("Hair Loss Prediction :bald_man:")
    st.subheader("Welcome to the Hair Loss Prediction App!")
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ20QikAaxXOrh2Td9muLkVcAOmrQfXNGwSXQ&usqp=CAU', width = 550)
    st.write("This app is designed to make the exploration and analysis of the Hair Loss Prediction Dataset easy and accessible. Whether you're interested in the distribution of data, relationships between features, or the performance of a machine learning model, this tool provides an interactive and visual platform for your exploration. With this model, you will be able to easily make predictions on wether an individual will experience hair loss or not.") 
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQURJP3PNhBtbn-ZOYixvqwq06x8FN0ZuU8EQ&usqp=CAU', width = 550)
    st.write("Use the sidebar to navigate between different sections.")

# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("The dataset is intended for exploratory data analysis, modeling, and predictive analytics tasks aimed at understanding the relationship between various factors and the likelihood of baldness in individuals. Within this dataset you will find information about various factors that may contribute to baldness in individuals. Each row represents a unique individual, and the columns represent different factors related to genetics, hormonal changes, medical conditions, medications and treatments, nutritional deficiencies, stress levels, age, poor hair care habits, environmental factors, smoking habits, weight loss, and the presence or absence of baldness.")
    st.image('data:image/jpeg;base64,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')
    st.link_button("Click Here for The Hair Loss Kaggle Dataset", "https://www.kaggle.com/datasets/amitvkulkarni/hair-health", help = None)



    st.subheader("Sneak Peak at the Data")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
    
    # Column List
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle("Further breakdown of columns"):
            num_cols = df.select_dtypes(include='number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols}\nObject Columns: {obj_cols}")

    # Shape
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

elif page == 'Exploratory Data Analysis':
    st.title("📊 Exploratory Data Analysis (EDA)")
     
    selected_col = st.selectbox('Select a column!', ['Genetics', 'Weight Loss ', 'Smoking', "Medical Conditions", 'Stress', 'Medications & Treatments', 'Poor Hair Care Habits ', 'Nutritional Deficiencies '])
    if selected_col: 
        grouped = df_eda.groupby([selected_col, 'Hair Loss']).size().unstack()
        grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100

        max_percent_hair_loss = grouped[1].groupby(selected_col).max()

        sorted_medical_conditions = max_percent_hair_loss.sort_values(ascending=False).index
        
        grouped_sorted = grouped.loc[sorted_medical_conditions]
     #   fig, ax = plt.subplots(figsize = (10,6))
        ax = grouped_sorted.plot(kind='bar', stacked=True, figsize=(10, 6))

        # Add labels to the bars
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1f}%', (x + width / 2, y + height / 2), ha='center', va='center')
        plt.title(f'{selected_col} & Hair Loss')
        plt.xticks(rotation = 90)
        fig = plt.gcf()    
        st.pyplot(fig)
            
        


            

if page == "Modeling":
    st.title(':gear: Modeling')
    st.markdown("On this page, you can see how well different *machine learning models* make predictions on the Hair Loss Dataset. The best machine learning model is the one that beats the baseline accuracy score which is 50.25%.")
    
    # Set up X and Y
    features = ['Genetics', 'Hormonal Changes', 'Age','Poor Hair Care Habits ', 'Environmental Factors', 'Smoking', 'Weight Loss ']
    X = df[features]
    y = df['Hair Loss']
    
    
    #Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    
    model_option = st.selectbox('Select a Model:', ['Logistic Regression', 'Random Forest', 'KNN'], index = None)
    
    if model_option: 
        if model_option == 'Logistic Regression':
            model = LogisticRegression()
        elif model_option == 'Random Forest':
            model = RandomForestClassifier()
        elif model_option == 'KNN':
            k_value = st.slider('Select a number of k', 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors =k_value, n_jobs = -1)
            
        if st.button("Let's see the performance!"): 
            
            
            model.fit(X_train_sc, y_train)
            
            # Display results
            st.subheader(f'{model} Evaluation:')
            st.text(f'Training Accuracy: {round(model.score(X_train_sc, y_train),3)}')
            st.text(f'Testing Accuracy: {round(model.score(X_test_sc, y_test),3)}')
            
            # Add section for Confusion Matrix
            ConfusionMatrixDisplay.from_estimator(model, X_test_sc, y_test, cmap = 'Blues')
            
            #This is turning confusion matrix onto display 'get current figure'
            cm_fig = plt.gcf()
            
            st.pyplot(cm_fig)
            
            st.write('Compared to our baseline of 50.25% we can see that our model beat the baseline accuracy score!')



if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Hair Loss!")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")
    
    genetics = st.slider('Genetics', 0.0, 1.0, 1.0, 1.0)
    horm_ch = st.slider('Hormonal Changes', 0.0, 1.0, 1.0, 1.0)
    age = st.slider("Age", 1.0, 100.0, 1.0, 1.0)
    poor_hair_care = st.slider('Poor Hair Care Habits ', 0.0, 1.0, 1.0, 1.0)
    envir_factors = st.slider('Environmental Factors', 0.0, 1.0, 1.0, 1.0)
    smoking = st.slider('Smoking', 0.0, 1.0, 1.0, 1.0)
    weight_loss = st.slider('Weight Loss ', 0.0, 1.0, 1.0, 1.0)
    
    user_input = pd.DataFrame({
            'Genetics': [genetics],
            'Hormonal Changes': [horm_ch],
            'Age': [age],
            'Poor Hair Care Habits ': [poor_hair_care],
            'Environmental Factors': [envir_factors],
            'Smoking': [smoking],
            'Weight Loss ': [weight_loss]
            })
    
    
    features = ['Genetics', 'Hormonal Changes', 'Age','Poor Hair Care Habits ', 'Environmental Factors', 'Smoking', 'Weight Loss ']
    X = df[features]
    y = df['Hair Loss']

    st.write("The predictions are made using KNN as it performed the best out of all of the models.")
    model = KNeighborsClassifier()
    
    if st.button("Make a Prediction!"):
        model.fit(X, y)
        prediction = model.predict(user_input)
        if prediction[0] == 1:
            st.write('The model predicts for hair loss!')
        else:
            st.write('The model predicts for no hair loss!')
        
        st.balloons()