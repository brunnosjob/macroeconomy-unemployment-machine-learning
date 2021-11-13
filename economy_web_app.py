#!/usr/bin/env python
# coding: utf-8

#Importando as bibliotecas
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
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
st.sidebar.markdown('Em breve haverá mais páginas com ricas informações. Aguarde!')
st.sidebar.markdown(' ')

#Redes sociais
st.sidebar.markdown('Feito por : Bruno Rodrigues Carloto')

st.sidebar.markdown("Redes Sociais :")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/bruno-rodrigues-carloto)")
st.sidebar.markdown("- [Medium](https://br-cienciadedados.medium.com)")
st.sidebar.markdown("- [Github](https://github.com/brunnosjob)")

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
pop = st.number_input('Insira a população do país:'0, 8000000000, 0)

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
