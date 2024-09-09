import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# CSS pour personnaliser le style
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .stButton>button {
        color: white;
        background-color: green;
        border-radius: 10px;
        padding: 10px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        border: 2px solid green;
        border-radius: 10px;
        padding: 5px; /* Augmenter le padding pour agrandir le cadre */
    }
    .header-image {
        width: 100%;
        height: auto;
    }
    /* Appliquer le style gras et bleu aux labels */
    label {
        font-weight: bold !important;
        color: green !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Charger les modèles et les encodeurs
model = joblib.load('C:\\Users\\pc\\OneDrive\\Bureau\\Oriflame\\ML1Oriflame\\bestmodel.pkl')
encoder = joblib.load('C:\\Users\\pc\\OneDrive\\Bureau\\Oriflame\\ML1Oriflame\\labelencoder.pkl')
model_2 = joblib.load('C:\\Users\\pc\\OneDrive\\Bureau\\Oriflame\\ML2Oriflame\\bestmodel2.pkl')
encoder_2 = joblib.load('C:\\Users\\pc\\OneDrive\\Bureau\\Oriflame\\ML2Oriflame\\labelencoder2.pkl')

# Section 1 : Prédiction de prix
st.image('p2.jpg', caption="Oriflame", use_column_width=True)
st.title('Prédiction de Prix Pour Les Produits Oriflame')

known_categories = encoder.classes_
rating = st.slider('Note du produit', min_value=0.0, max_value=5.0, step=0.1)
brand_name = st.selectbox('Nom de la marque', known_categories)

if brand_name not in known_categories:
    st.error(f'La marque {brand_name} n\'est pas reconnue. Veuillez en choisir une autre.')
else:
    encoded_brand = encoder.transform([brand_name])[0]
    input_data = pd.DataFrame([[rating, encoded_brand]], columns=['rating', 'brand_name_encoded'])
    input_data = input_data[['rating', 'brand_name_encoded']]
    if st.button('Prédire le prix'):
        try:
            predicted_price = model.predict(input_data)[0]
            st.write(f'Le prix prédit pour le produit avec une note de {rating} et de la marque {brand_name} est : {predicted_price:.2f} dhs ')
        except Exception as e:
            st.error(f'Une erreur est survenue : {e}')

# Section 2 : Prédiction de note d'évaluation
st.image('p1.jpg', caption="Oriflame", use_column_width=True)
st.title('Prédiction de la note d\'évaluation d\'un produit (Oriflame)')

known_categories_2 = encoder_2.classes_
product_price = st.number_input('Entrer le prix du produit en DHS ', min_value=0.0, step=1.0)
brand_name_2 = st.selectbox('Nom de la marque ', known_categories_2)

if brand_name_2 not in known_categories_2:
    st.error(f'La marque {brand_name_2} n\'est pas reconnue. Veuillez en choisir une autre.')
else:
    encoded_brand_2 = encoder_2.transform([brand_name_2])[0]
    input_data_2 = pd.DataFrame([[product_price, encoded_brand_2]], columns=['product_price', 'brand_name_encoded'])
    input_data_2 = input_data_2[['product_price', 'brand_name_encoded']]
    if st.button('Prédire la note d\'évaluation'):
        try:
            predicted_rating = model_2.predict(input_data_2)[0]
            st.write(f'La note prédite pour le produit avec un prix de {product_price} dhs et de la marque {brand_name_2} est : {predicted_rating:.2f} ')
        except Exception as e:
            st.error(f'Une erreur est survenue : {e}')

# Charger les données
data = pd.read_excel('C:\\Users\\pc\\OneDrive\\Bureau\\Oriflame\\ScrappingOriflame\\new.xlsx')

# Visualisation de la distribution des prix
st.subheader('Distribution des prix')
plt.figure(figsize=(10,6))
sns.histplot(data['product_price'], bins=30, kde=True , color='green')
plt.title("Distribution des prix")
plt.xlabel("Prix")
plt.ylabel("Fréquence")
plt.xticks(rotation=90)
st.pyplot(plt)

# La relation entre le prix et les notes
st.subheader('Relation entre le prix et les notes')
plt.figure(figsize=(12, 8))
sns.scatterplot(x='rating', y='product_price', data=data, s=50 , color='green')
plt.title('Relation entre le prix et les notes')
plt.xlabel('Note')
plt.ylabel('Prix')
plt.xticks(rotation=45)
st.pyplot(plt)

# La distribution des marques
st.subheader('Distribution des marques')
plt.figure(figsize=(15, 10))
sns.countplot(y='brand_name', data=data, order=data['brand_name'].value_counts().index , color='green')
plt.title('Distribution des marques')
plt.xlabel('Nombre de produits')
plt.ylabel('Marque')
plt.yticks(fontsize=12)
st.pyplot(plt)