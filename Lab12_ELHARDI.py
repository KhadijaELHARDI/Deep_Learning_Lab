# Lab12: Classification des fleurs iris en utilisant scikit-learn.
# Réalisé par ELHARDI Khadija EMSI 2023/2024
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd

# Step 1: DataSet
iris = datasets.load_iris()


# Step 2: Model
def choose_model(select_model, **kwargs):
    models = {
        'Random Forest': RandomForestClassifier,
        'Decision Tree': DecisionTreeClassifier,
        'KNN': KNeighborsClassifier,
        'SVM': SVC,
    }

    selected_model_class = models.get(select_model)

    if selected_model_class:
        # Instantiate the selected model with optional keyword arguments
        return selected_model_class(**kwargs)
    else:
        print(f"Le modèle spécifié ({select_model}) n'est pas reconnu.")
        return None


# Step 3: Training
def train_model(model, data, target):
    model.fit(data, target)
    return model


# Step 4: Test
def make_prediction(model, data):
    return model.predict(data)


st.header('Iris Flowers Classification')
st.image('images/iris.png')
st.sidebar.header('Iris Features')


def user_input():
    sepal_length = st.sidebar.slider('Sepal Length', 0.1, 7.9, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width', 0.1, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal Length', 0.1, 6.9, 1.0)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.5)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features


df = user_input()
st.write(df)
select_model = st.sidebar.selectbox("Select Your Learning Algorithm", ['Random Forest', 'Decision Tree', 'KNN', 'SVM'])
st.write('Selected Algorithm is: ' + select_model)
model = choose_model(select_model)
if model is not None:
    model = train_model(model, iris.data, iris.target)
    st.subheader('Prediction')
    prediction = make_prediction(model, df)
    st.write(iris.target_names[prediction])
    st.image('images/' + iris.target_names[prediction][0] + '.png')
else:
    st.write('Please select a valid algorithm.')



