#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import joblib
import streamlit as st
# Dividindo os tipos de dados para encaixar no streamlit
x_numericos = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 'extra_people': 0,
               'minimum_nights': 0, 'ano': 0,'qtd_amenities': 0, 'host_listings_count': 0, 'number_of_reviews': 0}

x_tf = {'is_business_travel_ready': 0, 'instant_bookable': 0}

x_listas = {'property_type': ['Apartment', 'Condominium', 'House', 'Guest suite', 'Hostel'],
            'room_type': ['Entire home/apt', 'Private room', 'Hotel room'],
            'cancellation_policy': ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period'],
            'bed_type': ['Airbed', 'Couch', 'Futon']
            }
#Criando dicionario separado para dar valores do x_listas de forma mais pratica
dicio = {}
for item in x_listas:
    for tipo in x_listas[item]:
        dicio[f'{item}_{tipo}'] = 0
# No x_numericos ( Definir 5 casas decimais para latitude e longiude, no extra people fiz o mesmo porém duas casas decimais... pro resto deixei valores inteiros)
for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format='%.5f')
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor
# no x_tf Fiz uma caixa com sim ou não que dependendo da resposta do usuario interpreta como 1 ou 0 pro modelo
for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor == 'Sim':
        x_tf[item] = 1
    else:
        x_tf[item] = 0
# No x_listas identifiquei um padrão de string e criei um dicionario que recebe as categorias(property_type,room_type, cancelation_policy) e com um '_'
# Ele bota as subcategorias de forma automatica com o for adicionando o valor/// Com isso o x_listas é usado apenas pro streamlit e o dicio pro modelo final
#
for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dicio[f'{item}_{valor}'] = 1
# Criando botão que será responsável pela previsão do valor do imovel usando o modelo que enviei pelo joblib
botao = st.button('Prever valor do Imóvel')
if botao:
    dicio.update(x_numericos)
    dicio.update(x_tf)
    df_valores = pd.DataFrame(dicio, index=[0])
    modelo = joblib.load(r'C:\Users\MASTER\Downloads\PROJETO CIENCIA DE DADOS\modelo1.joblib')
    preco = modelo.predict(df_valores)
    st.write(preco[0])
# OBS ( O Modelo deve está na mesma pasta que o arquivo.py do deploy para funcionar)


# In[ ]:





# In[ ]:




