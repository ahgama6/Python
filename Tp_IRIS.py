import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 1. Télécharger la base de données Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urllib.request.urlretrieve(url, 'iris.csv')
st.write("Base de données Iris téléchargée.")

# 2. Charger la base de données
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv('iris.csv', header=None, names=columns)
st.write("Base de données Iris chargée.")
st.write(iris_data.head())  # Afficher les premières lignes du DataFrame pour vérifier

# 3. Manipuler les données
# Afficher les statistiques descriptives des données
st.write("Statistiques descriptives :")
st.write(iris_data.describe())

# Filtrer les données pour la classe 'Iris-setosa'
setosa_data = iris_data[iris_data['class'] == 'Iris-setosa']

# Calculer la moyenne des longueurs de sépales pour chaque classe
mean_sepal_length = iris_data.groupby('class')['sepal_length'].mean()
st.write("Moyenne des longueurs de sépales pour chaque classe :")
st.write(mean_sepal_length)

# Ajouter une nouvelle colonne pour le rapport longueur/largeur des sépales
iris_data['sepal_ratio'] = iris_data['sepal_length'] / iris_data['sepal_width']

st.write("Manipulations des données terminées.")
st.write(iris_data.head())  # Afficher les premières lignes pour vérifier

# 4. Faire des graphes
# Histogramme de la longueur des sépales
st.write("Histogramme de la longueur des sépales :")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(iris_data['sepal_length'], bins=30, color='blue', edgecolor='black')
ax.set_title('Histogramme de la longueur des sépales')
ax.set_xlabel('Longueur des sépales')
ax.set_ylabel('Fréquence')
st.pyplot(fig)

# Graphique en ligne de la longueur et largeur des sépales
st.write("Longueur vs Largeur des Sépales :")
fig, ax = plt.subplots(figsize=(10, 6))
for species in iris_data['class'].unique():
    subset = iris_data[iris_data['class'] == species]
    ax.scatter(subset['sepal_length'], subset['sepal_width'], label=species)
ax.set_title('Longueur vs Largeur des Sépales')
ax.set_xlabel('Longueur des sépales')
ax.set_ylabel('Largeur des sépales')
ax.legend()
st.pyplot(fig)

# Box plot des longueurs de pétales par classe
st.write("Box plot des longueurs de pétales par classe :")
fig, ax = plt.subplots(figsize=(10, 6))
iris_data.boxplot(column='petal_length', by='class', grid=False, ax=ax)
ax.set_title('Box plot des longueurs de pétales par classe')
ax.set_xlabel('Classe')
ax.set_ylabel('Longueur des pétales')
fig.suptitle('')  # Supprimer le titre par défaut du suplot
st.pyplot(fig)
