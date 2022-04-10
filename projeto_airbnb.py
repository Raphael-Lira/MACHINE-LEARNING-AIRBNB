#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import pathlib 
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai')
from sklearn.metrics import r2_score,  mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import numpy as np 


# - Defenindo meses para poder colocar na base de dados 

# In[2]:


meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}   


# - Juntando arquivos e transformando em uma base de dados apenas
# * Usando o pathlib juntei todos os arquivos da pasta "dataset" logo em seguida tirei o ano e o mes com base no nome do arquivo que estava na pasta pra acrescentar dentro da base de dados e por ultimo juntei tudo e fiz a base de dados completa com o nome de "leitura_dados"

# In[41]:


caminhobase = pathlib.Path('dataset')
base_dados = pd.DataFrame()
for arquivo in caminhobase.iterdir():
    mes = meses[arquivo.name[:3]]#utilizando o proprio nome do arquivo para definir uma coluna de mês e ano
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))     
    df = pd.read_csv(caminhobase / arquivo.name)
    df['mes'] = mes
    df['ano'] = ano
    base_dados = base_dados.append(df)
display(base_dados)


# - Criando a base de dados em um excel com apenas 1000 linhas para poder entender cada coluna para começar a limpeza
# - Após criar a base de dados em excel começei a excluir todas as informações que não são necessarias para tirar um valor 
# - Exemplo: IDs, nomes, descrições abertas, informações repetidas, colunas sem valores ou iguais

# In[4]:


base_dados.head(1000).to_csv(r'C:\Users\MASTER\Downloads\PROJETO CIENCIA DE DADOS\leitura_dados.csv', sep=';')


# - No excel eu usei um metodo pra pegar os nomes das colunas e colocar uma virgula automaticamente e trouxe pro codigo de maneira simples e sem ter trabalho de escrever todas as colunas manualmente, depois dissso resumi a coluna apenas as colunas que na minha analise seriam úteis pro codigo

# In[40]:


colunas = ['host_response_time','host_response_rate','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee', 'guests_included', 'extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano']
base_dados = base_dados[colunas]
display(base_dados)


# - Enviando base de dados limpa para rodar de forma mais leve no PC caso eu tenha que reniciar o jupyter

# In[12]:


base_dados.to_csv(r'C:\Users\MASTER\Downloads\PROJETO CIENCIA DE DADOS\base_limpa.csv', sep=';')


# - Tratando valores vazios 
# - 1: Todas as colunas que tinham mais de 250000 valores vazios foram excluidas ( não só por ter valores vazios mas por não serem tão relevantes para o objetivo)
# - 2: Usei o isnull().sum() para me auxiliar

# In[10]:


for coluna in base_dados:
    if base_dados[coluna].isnull().sum() > 250000:
        base_dados = base_dados.drop(coluna, axis='columns')


# In[21]:


base_dados = base_dados.dropna(subset = ['bathrooms', 'bedrooms', 'beds', 'host_listings_count'])


# In[23]:


base_dados.isnull().sum()


# - Mudando tipos de dados para partir para o modelo de previsão 
# - 1: inicialmente usei o .info usando o parametro verbose para saber qual tipo de cada coluna
# - 2: Em seguinda usei o iloc[0] pegando apenas a primeira informação para ir confirmando cada informação
# - 3: Em seguida tanto o 'price' e o 'extra_people' estavam como um object então eu tratei ele como str tirando o '$' e as virgulas e em seguida transformei em float com o astype

# In[10]:


print(base_dados.iloc[0])


# In[16]:


#3
#price , extra_people(colunas que estão com valores em string sendo que deveriam ser float)
base_dados['price'] = base_dados['price'].str.replace('$', '')
base_dados['price'] = base_dados['price'].str.replace(',', '')
base_dados['price'] = base_dados['price'].astype(float, errors='raise')
#3
base_dados['extra_people'] = base_dados['extra_people'].str.replace('$', '')
base_dados['extra_people'] = base_dados['extra_people'].str.replace(',', '')
base_dados['extra_people'] = base_dados['extra_people'].astype(float, errors='raise')


# - Vendo a correlação e plotando com o seanor 

# In[7]:


plt.figure(figsize=(20,15))
sns.heatmap(base_dados.corr(), annot=True, cmap='Blues')


# - Criando graficos e funções para saber o limite inferior e superior de forma mais precisa 
#  e saber se vamos tirar certos outliers pelos graficos

# In[36]:


def limite(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = (q3 - q1)
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude
print(limite(base_dados['price']))


# In[34]:


def diagrama(coluna):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    sns.boxplot(x=coluna, ax= ax1)
    ax2.set_xlim(limite(coluna))
    sns.boxplot(x=coluna, ax=ax2)


# In[6]:


def histograma(coluna):
    plt.figure(figsize=(15,5))
    sns.distplot(coluna, hist=True)


# In[7]:


def grafico_barra(coluna): # O grafico de barra foi feito para a exclusão mais precisa de valores menores
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index,y= coluna.value_counts())
    ax.set_xlim(limite(coluna))


# - Ultilizando os graficos em cada coluna para ter certeza na exclusão dos outliers

# ## Coluna Price
# - Criando graficos para ver outliers de forma mais precisa

# In[34]:


diagrama(base_dados['price'])
histograma(base_dados['price'])


# ## Coluna extra_people

# In[33]:


diagrama(base_dados['extra_people'])
histograma(base_dados['extra_people'])


# ## Host_list_count 

# In[32]:


diagrama(base_dados['host_listings_count'])
grafico_barra(base_dados['host_listings_count'])


# ## Coluna accommodates

# In[31]:


diagrama(base_dados['accommodates'])
grafico_barra(base_dados['accommodates'])


# ## Coluna bedroomns

# In[30]:


diagrama(base_dados['bedrooms'])
grafico_barra(base_dados['bedrooms'])


# ## Coluna bathrooms

# In[29]:


diagrama(base_dados['bathrooms'])
grafico_barra(base_dados['bathrooms'])


# ## Coluna beds

# In[39]:


diagrama(base_dados['beds'])
grafico_barra(base_dados['beds'])


# ## Coluna guests_included       
# - Eu optei pela exclusão da coluna guest_included para evitar um possivel erro no meu modelo...a maior parte dos valores está em 1 oque indica ser algum tipo de erro por que não faz sentido ter apenas um...

# In[47]:


diagrama(base_dados['guests_included'])
grafico_barra(base_dados['guests_included'])
base_dados = base_dados.drop('guests_included', axis='columns')


# ## Coluna number_of_reviews
# - Mantive o number_of_reviews por que acredito seja importante os outliers

# In[58]:


diagrama(base_dados['number_of_reviews'])
grafico_barra(base_dados['number_of_reviews'])


# ## Coluna minimum_nights

# In[57]:


diagrama(base_dados['minimum_nights'])
grafico_barra(base_dados['minimum_nights'])


# ## Coluna maximum_nights
# - Assim como na coluna guests_included eu exclui a coluna maximum nights por ser na minha opinão valores que não acrescentar no codigo ou fazer alguma diferença

# In[56]:


diagrama(base_dados['maximum_nights'])
grafico_barra(base_dados['maximum_nights'])
base_dados = base_dados.drop('maximum_nights', axis='columns')


# ## Exclusão de Outliers
# - Excluindo todos os outliers que no objetivo do projeto ira atrapalhar...
# - Exclui da coluna Price por que o modelo de previsão é para imoveis de valor comum
# - Exclui da coluna extra_people os outliers por que para ter espaços para muitas pessoas o preço muda totalmente por causa do espaço a mesma -  - razão vai para as outas colunas (host_listings_count, accommodates, beds, bedsroom e etc)

# In[5]:


def excluir_outliers(df, coluna):
    lim_inf, lim_sup = limite(df[coluna])
    filtro =  df[coluna] < lim_sup
    filtrado = df[filtro]
    return filtrado


# In[52]:


base_dados = excluir_outliers(base_dados, 'extra_people')
base_dados = excluir_outliers(base_dados, 'price')
base_dados = excluir_outliers(base_dados, 'host_listings_count')
base_dados = excluir_outliers(base_dados, 'minimum_nights')
base_dados = excluir_outliers(base_dados, 'number_of_reviews')
base_dados = excluir_outliers(base_dados, 'bedrooms')
base_dados = excluir_outliers(base_dados, 'beds')
base_dados = excluir_outliers(base_dados, 'accomodates')
base_dados = excluir_outliers(base_dados, 'bathrooms')


# ## Tratando valores de texto
# - property_type                                                     
# - room_type 
# - bed_type                                                            
# - amenities
# - cancellation_policy 

# ## property_type
# - Na coluna tinha variados tipos de propriedade abaixo de 2000 quantidades para deixar o dataframe mais conciso coloquei todas as categorias abaixo de 2000 como 'Other' 

# In[4]:


for tipo, quantidade in base_dados['property_type'].value_counts().items():
    if quantidade < 2000:
        base_dados.loc[base_dados['property_type'] == tipo, 'property_type'] = 'Other'
print(base_dados['property_type'].value_counts())


# ## room_type, bed_type
# - Na colunas room_type,bed_type não fiz alteração nenhuma por quê não achei nescessário 

# In[5]:


print(base_dados['room_type'].value_counts())
print(base_dados['bed_type'].value_counts())


# ## cancellation_policy

# In[6]:


for tipo, quantidade in base_dados['cancellation_policy'].value_counts().items():
    if quantidade < 10000:
        base_dados.loc[base_dados['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict'
print(base_dados['cancellation_policy'].value_counts())


# ## amenities
# - A coluna amenities tem varios valores em um campo e extrair esses valores de forma dinamica vai pesar muito no codigo
# por tanto eu preferi contar quantos amenities tem e dai tirar sua importancia pra cada imovel, além de excluir seus outliers para evitar possiveis

# In[7]:


base_dados['qtd_amenities'] = base_dados['amenities'].str.split(',').apply(len)
diagrama(base_dados['qtd_amenities'])
base_dados = excluir_outliers(base_dados, 'qtd_amenities')
base_dados = base_dados.drop('amenities', axis='columns')


# ## Encoding
# - Para que meu modelo de previsão funcione preciso reconfigurar minhas features com valores de texto
# - Para as colunas de True/False irei por 1= True / 2= False

# In[8]:


base_dados['instant_bookable'] = base_dados['instant_bookable'].map({'t':1, 'f':0}) 
base_dados['is_business_travel_ready'] = base_dados['is_business_travel_ready'].map({'t':1, 'f':0}) 


# In[9]:


base_dados_cod =pd.get_dummies(data= base_dados, columns=['cancellation_policy', 'property_type', 'room_type', 'bed_type'])
display(base_dados_cod.head())


# ## Escolhendo modelo
# * Escolhendo o modelo apartir de uma função que ira me retornar o R2 e o RSME para decidir com clareza

# In[24]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo} / R2 {r2:.2%} / RSME{RSME:.2f}'


# * Escolhi 3 modelos conhecidos entre variados que existem e coloquei dentro de um dicionario para fazer um for e ter a resposta dos 3 de uma vez

# In[23]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()
modelos = {'RandomFortes': modelo_rf,
           'LinearRegression': modelo_lr, 
           'ExtraTrees': modelo_et,
          }
#y = base_dados_cod['price'] 
#X = base_dados_cod.drop('price', axis='columns')


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)


# In[13]:


for nome_modelo, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ## O modelo escolhido foi o ExtraTrees
# * Para ter uma visualização melhor do ExtraTrees criei em ordem decrescente uma tabela que me da a importancia de cada coluna

# In[23]:


print(modelo_et.feature_importances_)
print(X_train.columns)
importancia_feature = pd.DataFrame(modelo_et.feature_importances_, X_train.columns)
importancia_feature = importancia_feature.sort_values(by=0, ascending=False)
display(importancia_feature)


# ## Finalizando o modelo tirando as colunas que não tiveram utilidade no calculo
# * Na primeira linha está a exclusão de todas as colunas que eu acredito que não estavam fazendo alguma diferença significativa e quando retirei e fiz o mesmo teste teve uma diferença insignificante levando em consideração a quantidade de colunas que retirei

# In[16]:


base_dados_cod = base_dados_cod.drop(['property_type_Bed and breakfast', 'room_type_Shared room', 'bed_type_Real Bed', 'property_type_Serviced apartment', 'bed_type_Pull-out Sofa', 'property_type_Loft', 'property_type_Other', 'mes'], axis='columns')
y = base_dados_cod['price']
X = base_dados_cod.drop('price', axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao)) 


# * Só fiquei com essas colunas 
# 
# 'host_listings_count', 'latitude', 'longitude', 'accommodates',
#        'bathrooms', 'bedrooms', 'beds', 'extra_people', 'minimum_nights',
#        'number_of_reviews', 'instant_bookable', 'is_business_travel_ready',
#        'ano', 'qtd_amenities', 'cancellation_policy_flexible',
#        'cancellation_policy_moderate', 'cancellation_policy_strict',
#        'cancellation_policy_strict_14_with_grace_period',
#        'property_type_Apartment', 'property_type_Condominium',
#        'property_type_Guest suite', 'property_type_Hostel',
#        'property_type_House', 'room_type_Entire home/apt',
#        'room_type_Hotel room', 'room_type_Private room', 'bed_type_Airbed',
#        'bed_type_Couch', 'bed_type_Futon', 'price'
# 
# E o Modelo ExtraTrees deu o seguinte resultado (R2 97.50% / RSME41.55)

# ## Deploy do projeto
# * Usei o joblib por suas recomendações

# In[19]:


import joblib
X['price'] = y
X.to_csv('base_final.csv')
joblib.dump(modelo_et, 'modelo1.joblib')


# # Tradução do projeto airbnb pra concluir tudo 

# In[30]:


import pandas as pd 
df = pd.read_csv('base_final.csv')
df = df.drop('Unnamed: 0', axis='columns')
print(df.columns)


# ## Renomeando todas as colunas para ficar em português no streamlit

# In[29]:


df.rename(columns = {'host_listings_count': 'Quantos imóveis no airbnb', 'bathrooms': 'Quantos banheiros', 'accommodates': 'Quantas pessoas cabem na casa', 'bedrooms': 'Quantos quartos',
                    'beds': 'Quantidade de camas disponiveis', 'extra_people': 'Preço por pessoas extras', 'minimum_nights': 'Noites minimas', 'number_of_reviews': 'Numero de avaliações',
                    'instant_bookable': 'Reserva instantânea', 'is_business_travel_ready': 'Disponivel pra reserva', 'ano': 'Ano', 'qtd_amenities': 'Quantidade de serviços (Media de serviços por apartamento é de 10 a 20)', 'cancellation_policy_strict': 'Sem cancelamento de hospedagem', 'cancellation_policy_flexible': 'Cancelamento de hospedagem flexível',
                    'cancellation_policy_moderate': 'Cancelamento de hospedagem moderado', 'cancellation_policy_strict_14_with_grace_period': 'Sem cancelamento de hospedagem por um periodo', 'property_type_Apartment': 'Apartamento', 'property_type_Condominium': 'Condominio', 'property_type_Guest suite': 'Suíte de hóspedes', 'property_type_Hostel': 'Hostel', 'property_type_House': 'Casa',
                    'room_type_Entire home/apt': 'Quarto inteiro', 'room_type_Hotel room': 'Quarto de hotel', 'room_type_Private room': 'Quarto privado', 'bed_type_Airbed': 'Cama de ar', 'bed_type_Couch': 'Colchão', 'bed_type_Futon': 'Sófa cama', 'price': 'Preço'}, inplace = True)
df.rename(columns = {'Cancelamento de hospedagem flexível': 'Flexível', 'Cancelamento de hospedagem moderado': 'Moderado', 'Sem cancelamento de hospedagem': 'Restrito', 'Sem cancelamento de hospedagem por um periodo': 'Restrito por um periodo'}, inplace=True)
print(df.columns)


# ## Finalizando com o treinamento do modelo em português

# In[27]:


import joblib
y = df['Preço']
X = df.drop('Preço', axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao)) 
joblib.dump(modelo_et, 'modelo2.joblib')

