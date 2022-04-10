#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import joblib
import streamlit as st
# Dividindo os tipos de dados para encaixar no streamlit
x_numericos = {'latitude': 0, 'longitude': 0, 'Quantas pessoas cabem na casa': 0, 'Quantos banheiros': 0, 'Quantos quartos': 0, 'Quantidade de camas disponiveis': 0, 'Preço por pessoas extras': 0,
               'Noites minimas': 0, 'Ano': 0,'Quantidade de serviços (Media de serviços por apartamento é de 10 a 20)': 0, 'Quantos imóveis no airbnb': 0, 'Numero de avaliações': 0}

x_tf = {'Reserva instantânea': 0, 'Disponivel pra reserva': 0}

x_listas = {'Tipo de imóvel ': ['Apartamento', 'Condominio', 'Casa', 'Suíte de hóspedes', 'Hostel'],
            'Tipo de quarto': ['Quarto inteiro', 'Quarto privado', 'Quarto de hotel'],
            'Politica de cancelamento ': ['Flexível', 'Moderado', 'Restrito', 'Restrito por um periodo'],
            'Tipo de cama': ['Cama de ar', 'Colchão', 'Sófa cama']
            }
#Criando dicionario separado para dar valores do x_listas de forma mais pratica
dicio = {}
for item in x_listas:
    for tipo in x_listas[item]:
        dicio[f'{tipo}'] = 0
# No x_numericos ( Definir 5 casas decimais para latitude e longiude, no extra people fiz o mesmo porém duas casas decimais... pro resto deixei valores inteiros)
for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format='%.5f')
    elif item == 'Preço por pessoas extras':
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
    dicio[f'{valor}'] = 1
# Criando botão que será responsável pela previsão do valor do imovel usando o modelo que enviei pelo joblib
botao = st.button('Prever valor do Imóvel')
if botao:
    dicio.update(x_numericos)
    dicio.update(x_tf)
    df_valores = pd.DataFrame(dicio, index=[0])
    modelo = joblib.load(r'C:\Users\MASTER\Downloads\PROJETO CIENCIA DE DADOS\modelo2.joblib')
    preco = modelo.predict(df_valores)
    st.write(f'A previsão do valor do imóvel foi de R${preco[0]:.2f}')
# OBS ( O Modelo deve está na mesma pasta que o arquivo.py do deploy para funcionar)

