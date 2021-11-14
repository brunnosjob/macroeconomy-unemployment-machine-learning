#!/usr/bin/env python
# coding: utf-8

#Importando as bibliotecas
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from PIL import Image
import pickle


#Buscando modelo de previsão de Saúde Fiscal
with open('simple_health_fiscal_tree.pkl', 'rb') as f:
    tree_model_fiscal = pickle.load(f)
    
#Buscando modelo de previsão de taxa de desemprego
with open('unemployment_simple_lm.pkl', 'rb') as f:
    regressor_sim_log = pickle.load(f)


#Construindo Web App
#Criando sidebar
#Orientando visão
st.markdown('*__Observação: para mais informações acerca do projeto, clique na seta no canto esquerdo superior da tela__*')
st.markdown(' ')

#Informações em sidebar
#Sobre o artigo
st.sidebar.subheader('Projeto de portfólio de Ciência de Dados')
st.sidebar.markdown('Em breve haverá o artigo descrevendo o passo a passo do desenvolvimento do projeto. Aguarde!')
st.sidebar.markdown(' ')

#Menu
st.sidebar.title('Menu')
pag = st.sidebar.selectbox('Selecione a página:', ['Interagir com o modelo', 'Sobre o conjunto de dados', 'Sobre os modelos'])
st.sidebar.markdown(' ')

#Redes sociais
st.sidebar.markdown('Feito por : Bruno Rodrigues Carloto')

st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")

#Abrindo as páginas
#Página 1
if pag == 'Interagir com o modelo':

    #Criação da página de interação com o(s) modelo(s)
    #Boas-vindas
    st.header('Bem-vindo à 4Economy')
    st.subheader('Uma aplicação web de machine learning voltada para a macroeconomia')
    st.markdown(' ')

    #Explicações
    st.markdown('### Como você pode experimentar e usar essa aplicação:')
    st.markdown('''1 - Você deve preencher o espaço com um valor percentual de 0 a 100.
    Esse valor está em porcentagem. Ele representa, em termos percentuais,
    o quanto do PIB de um país está comprometido com a dívida pública desse país;

    1 - Você pode simplesmente inserir dados fictícios, dentro dos limites estabelecidos para a aplicação;

    2 - Você pode realizar uma pesquisa e usar dados reais:

    I - Escolha um país;

    II - Pesquise sobre o quanto do PIB desse país está comprometido com sua dívida pública.
    O valor deve estar em porcentagem, quando for colocado no espaço de preenchimento, presente logo abaixo.

    III - Compare o resultado da predição acerca da taxa de desemprego com o valor real da taxa de desemprego
    desse país.

    3 - Você pode repetir o processo inúmeras vezes, inclusive, usar para algum trabalho ou projeto.''')
    st.markdown(' ')

    #Buscando modelo de predição de saúde fiscal
    with open('simple_health_fiscal_tree.pkl', 'rb') as f:
        tree_model_fiscal = pickle.load(f)
    
    #Buscando modelo de previsão de taxa de desemprego
    with open('unemployment_simple_lm.pkl', 'rb') as f:
        regressor_sim_log = pickle.load(f)

    #Aplicação dos modelos
    #Predição de saúde fiscal
    #Nome do país
    pais = st.text_input('Insira o nome do pais:')

    #População
    pop = st.number_input('Insira a população do país:', 0, 8000000000, 0)

    #Valor percentual da dívida pública
    public_debt = st.number_input('Insira o percentual (%) do PIB comprometido com a dívida pública:', 0.0, 100.0, 0.0)

    #Transformações e predição
    debt = np.array(public_debt)
    debt = debt.reshape(-1,1)
    predicted_health_fiscal = tree_model_fiscal.predict(debt)

    #Predição de taxa de desemprego
    #Transformações e predição
    X_log = np.log(predicted_health_fiscal)
    X_log = X_log.reshape(-1,1)
    unemployment = regressor_sim_log.predict(X_log)
    unemployment = np.round(unemployment, 3)
    taxa = unemployment/100
    desempregado = (pop * taxa)

    st.write('País:', pais)
    st.write('A taxa de desemprego está em torno de {}%.'.format(unemployment))
    st.write('Aproximação do total de desempregado do país:', desempregado)


    
#Página 2
if pag == 'Sobre o conjunto de dados':
    st.title('Qualidade dos dados')
    st.markdown('''
    Inicialmente, o modelo continha 56 linhas, referentes a 56 países, e 44 colunas. As colunas apresentavam informações
    econômicas, de desempenho jurídico e de desempenho burocrático e de relação com a língua inglesa. Há países de todos os continentes e de diferentes desenvolvimento.
    ''')
    st.markdown('#### Correlações antes da retirada de outliers')
    st.markdown('Antes de a retirada de outliers, as principais correlações com as variáveis Fiscal Health (saúde fiscal) e Unemployment (taxa de desemprego) eram:')
    corr_min = Image.open('corr_min.png')
    st.image(corr_min, use_column_width=True)
    st.markdown('''
    As principais correlações eram:
    
    #### Para com a variável dependente Fiscal Health:
    
    Public Debt (% of GDP): -0,51;
    
    Trade Freedom: 0,6;
    
    Unemployment (%): -0,58.
    
    #### Para com a variável dependente Unemployment (%):
    
    Inflation (%): 0,57;
    
    Fiscal Health: -0,58;
    
    GDP Growth Rate (%): -0,63.
    ''')
    st.markdown('Após a retirada de outliers, as principais correlações com as variáveis Fiscal Health (saúde fiscal) e Unemployment (taxa de desemprego) eram:')
    sem_out_corr = Image.open('sem_out_corr.png')
    st.image(sem_out_corr, use_column_width=True)
    
#Página 3
if pag == 'Sobre os modelos':
    st.title('Qualidade dos modelos')
    st.markdown('###### Modelo para estimativa da saúde fiscal a partir do percentual do PIB comprometido para com a dívida pública')
    st.markdown(' ')
    total_corr = Image.open('total_corr.png')
    st.image(total_corr, use_column_width=True)
    st.markdown('''Observando o gráfico acima, pode-se extrair as correlações entre todas as variáveis numéricas da base de dados, além das duas variáveis
    dependentes dos respectivos modelos.''')
    st.markdown(' ')
                
    
    
    st.markdown(' ')
