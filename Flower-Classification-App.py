import streamlit as st
from sklearn import svm  
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Title
st.title("Iris Flower Classification App")

# Create Data
data = "Iris.csv"


# To Improve speed and cache data


@st.cache(persist=True)
def explore_data(data):
    df = pd.read_csv(data)
    return df


df = explore_data(data)

# show data frame using head and tail in pandas.
st.subheader('Preview Data')
if st.checkbox('Top 5 rows'):
    st.write(df.head())
elif st.checkbox('Bottom 5 rows'):
    st.write(df.tail())
else:
    st.write(df.head(10))

st.subheader('Summary of Data frame')
if st.checkbox("Show all column names"):
    st.write(df.columns)
elif st.checkbox("Statistical Summary"):
    st.write(df.describe())
else:
    st.write(df.shape)

st.subheader('Show all plots')
st.text("Pairplot")
pairplot = sns.pairplot(df, vars=["SepalLengthCm", "SepalWidthCm",
                                  "PetalLengthCm", "PetalWidthCm"], hue="Species")
st.pyplot(pairplot)


st.text("Bar chart on Species and Petal width")
bar_chart, ax = plt.subplots()
sns.barplot(df["Species"], df["PetalWidthCm"], ax=ax)
# bar_chart = sns.barplot(x='Species', y='PetalWidthCm', hue='Species', data=df)
st.pyplot(bar_chart)

st.text("Correlation Matrix")
heat_map, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True)
st.write(heat_map)

X = df.iloc[:, 1:5]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

classifier = svm.SVC()
scaler = StandardScaler()
scaler.fit(X_train)
classifier.fit(X_train, y_train)


# predicted_values = classifier.predict(x_test)


# st.write(df[df.Species == "Iris-versicolor"])


def Input_Output():
    #st.sidebar.header('Input Features')
    sepal_length = st.slider(
        label='Sepal Length',
        min_value=df['SepalLengthCm'].min(),
        max_value=df['SepalLengthCm'].max(),
        value=float(df['SepalLengthCm'].mean()),
        step=0.1)
    
    sepal_width = st.slider(
        label='Sepal Width',
        min_value=df['SepalWidthCm'].min(),
        max_value=df['SepalWidthCm'].max(),
        value=float(df['SepalWidthCm'].mean()),
        step=0.1)
    
    petal_length = st.slider(
        label='Petal Length',
        min_value=df['PetalLengthCm'].min(),
        max_value=df['PetalLengthCm'].max(),
        value=float(df['PetalLengthCm'].mean()),
        step=0.1)
    
    petal_width = st.slider(
        label='Petal Width',
        min_value=df['PetalWidthCm'].min(),
        max_value=df['PetalWidthCm'].max(),
        value=float(df['PetalWidthCm'].mean()),
        step=0.1)

    X_scaled = scaler.transform(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_values = classifier.predict(X_scaled)
    
    #st.write(predicted_values)
    
    if predicted_values == "Iris-virginica":
        iris_virginica_image = Image.open('Virginica.jpg')
        st.image(iris_virginica_image, "Virginica")
        
    elif predicted_values == "Iris-versicolor":
        iris_versicolor_image = Image.open('Versicolor.jpg')
        st.image(iris_versicolor_image, "Versicolor")
        
    elif predicted_values == "Iris-setosa":
        iris_setosa_image = Image.open('Setosa.jpg')
        st.image(iris_setosa_image, "Setosa")
    
        
        
Input_Output()
