import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE 
import umap.umap_ as umap
from umap import UMAP
from math import dist
from PIL import Image


# Logo image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://tenor.com/view/lillee-jean-style-your-lillee-style-makeup-beauty-gif-25713077.gif");
             background-attachment: fixed;
             background-size: cover
         }}
         [data-testid='stHeader'] {{
             background-color: rgba(0,0,0,0);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.markdown('<style>h1{color: #e60073;}</style>', unsafe_allow_html=True)
st.title('Skincare Sifarish')


#st.write("Welcome to your skincare products recommendation engine!:")
st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Welcome to your skincare products recommendation engine"}</h1>', unsafe_allow_html=True)

#Load the data
cosmeticsdf = pd.read_csv('cosmetics.csv')

cosmeticsdf = cosmeticsdf[cosmeticsdf["Ingredients"].str.contains("Visit") == False]
cosmeticsdf = cosmeticsdf[cosmeticsdf["Ingredients"].str.contains("No Info") == False]
cosmeticsdf = cosmeticsdf[cosmeticsdf["Ingredients"].str.contains("NAME") == False]
cosmeticsdf = cosmeticsdf[cosmeticsdf["Ingredients"].str.contains("product package") == False]


# Choose a product category
st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Select the product category you are looking for:"}</h1>', unsafe_allow_html=True)

category = st.selectbox(label='', options= cosmeticsdf['Label'].unique() )
category_subset = cosmeticsdf[cosmeticsdf['Label'] == category]

# Choose a brand
st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Select a brand you love:"}</h1>', unsafe_allow_html=True)

brand = st.selectbox(label='', options= sorted(category_subset['Brand'].unique()))
category_brand_subset = category_subset[category_subset['Brand'] == brand]


# Choose product
st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Select the product:"}</h1>', unsafe_allow_html=True)

product = st.selectbox(label='', options= sorted(category_brand_subset['Name'].unique() )) 

# Choose skin type

#st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Select your skin type"}</h1>', unsafe_allow_html=True)
#skin_type = st.selectbox(label='', options= ['Combination','Dry', 'Normal', 'Oily', 'Sensitive'] )

            

CD = category_subset.copy()


def one_hot_encoder(tokens):
    x = np.zeros(N)
    
    for ingredient in tokens:
        index = ingredient_dict[ingredient]
        x[index] = 1
    return x

if category is not None:
    category_subset = cosmeticsdf[cosmeticsdf['Label'] == category]
    
if product is not None:
    category_subset = category_subset.reset_index(drop=True)
    
    
    
# tokenisation of the ingredients list 
index = 0
ingredient_dict = {}
corpus = []

for i in range(len(category_subset)):
    ingredients = category_subset['Ingredients'][i]
    ingredients_lower = ingredients.lower()        # change all to lower case
    tokens = ingredients_lower.split(', ')         # split up the ingredients from the string
    corpus.append(tokens)
    
    for ingredient in tokens:
        if ingredient not in ingredient_dict:      # prevents duplication
            ingredient_dict[ingredient] = index
            index += 1
            
# matrix that is filled with binary values 
# to check if ingredient is present or absent
# if present, it will be 1
# if absent, it will be 0
 
M = len(category_subset)           # number of products
N = len(ingredient_dict)    # number of ingredients

# initialise matrix with 0s

matrix = np.zeros(shape = (M, N))
   
i = 0
for tokens in corpus:
    matrix[i, :] = one_hot_encoder(tokens)
    i += 1
    
model_run = st.button('Find similar products!')


if model_run:

    #st.write('Based on the ingredients of the product you selected')
    st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"Based on the ingredients of the product you selected"}</h1>', unsafe_allow_html=True)
    
    #st.write('here are the top 6 products that are the most similar:')
    st.markdown(f'<h1 style="color:#ff3399;font-size:24px;">{"here are the top 5 products that are the most similar:"}</h1>', unsafe_allow_html=True)

    
# dimensionality reduction with t-SNE

# Dimension reduction with t-SNE
#tSNE_data = TSNE(n_components = 2, learning_rate = 200).fit_transform(matrix)
#st.write(tSNE_data.shape)

# Make X, Y columns 
#category_subset['X'] = tSNE_data[:, 0]
#category_subset['Y'] = tSNE_data[:, 1] 
 
# dimension reduction with UMAP
    umap_data = umap.UMAP(n_components = 2, min_dist = 0.7, n_neighbors = 1000, random_state = 1).fit_transform(matrix)

    # adding 2 new columns X and Y to the dataset

    category_subset['X'] = umap_data[:, 0]
    category_subset['Y'] = umap_data[:, 1]

    category_subset['Distance'] = 0.0


    myItem = category_subset[category_subset['Name'] == product]

    point1 = np.array([myItem['X'], myItem['Y']])


    # other items

    for i in range(len(category_subset)):
        point2 = np.array([category_subset['X'][i], category_subset['Y'][i]])
        category_subset.Distance[i] = dist(point1, point2)

    # sorting data in ascending order

    #category_subset = category_subset.sort_values('Distance')
    #category_subset.head(6)

    # arrange by descending order
    top_picks = category_subset.sort_values(by=['Distance'])

     # Select relevant columns
    top_picks = top_picks[['Label', 'Brand', 'Name', 'Price','Rank']]
    top_picks = top_picks.reset_index(drop=True)
    top_picks = top_picks.drop(top_picks.index[0])

    st.dataframe(top_picks.head(5))
