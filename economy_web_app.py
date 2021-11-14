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
    st.markdown('''
    
    1 - Você deve preencher o espaço com um valor percentual de 0 a 100.
    Esse valor está em porcentagem. Ele representa, em termos percentuais,
    o quanto do PIB de um país está comprometido com a dívida pública desse país;

    2 - Você pode simplesmente inserir dados fictícios, dentro dos limites estabelecidos para a aplicação;

    3 - Você pode realizar uma pesquisa e usar dados reais:

    I - Escolha um país;

    II - Pesquise sobre o quanto do PIB desse país está comprometido com sua dívida pública.
    O valor deve estar em porcentagem, quando for colocado no espaço de preenchimento, presente logo abaixo.

    III - Compare o resultado da predição acerca da taxa de desemprego com o valor real da taxa de desemprego
    desse país.

    4 - Você pode repetir o processo inúmeras vezes, inclusive, usar para algum trabalho ou projeto.''')
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
    
    
    Os dados não apresentavam ausência de valores tampouco inconsistência de valores, como idades negativas e/ou alturas negativas, informações irreais.''')
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
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('#### Correlações após retirada de outliers')
    
    
    st.markdown('Após a retirada de outliers, as principais correlações com as variáveis Fiscal Health (saúde fiscal) e Unemployment (taxa de desemprego) eram:')
    sem_out_corr = Image.open('sem_out_corr.png')
    st.image(sem_out_corr, use_column_width=True)
    st.markdown('''
    As principais correlações eram:
    
    #### Para com a variável dependente Fiscal Health:
    
    Public Debt (% of GDP): -0,58;
    
    Unemployment (%): -0,68.
    
    #### Para com a variável dependente Unemployment (%):
    
    Public Debt (% of GDP): 0,51
    
    Fiscal Health: -0,68;
    ''')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('''Como pode-se apreender, houve potencialização ente certas variáveis independentes e as variáveis dependentes, enquanto, 
    em relação a outras independentes, houve enfraquecimento correlacional.''')
    
    st.markdown(' ')
    st.markdown(' ')
    
    st.markdown('#### Gráficos de caixa evidenciando outliers nas principais variáveis para o teste e desenvolvimento do modelo')
    total_outlier0 = Image.open('total_outlier0.png')
    st.image(total_outlier0, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier1 = Image.open('total_outlier1.png')
    st.image(total_outlier1, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier2 = Image.open('total_outlier2.png')
    st.image(total_outlier2, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier3 = Image.open('total_outlier3.png')
    st.image(total_outlier3, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier4 = Image.open('total_outlier4.png')
    st.image(total_outlier4, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier5 = Image.open('total_outlier5.png')
    st.image(total_outlier5, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    total_outlier6 = Image.open('total_outlier6.png')
    st.image(total_outlier6, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')

    st.markdown('#### Gráficos de caixa evidenciando nova configuração da posição dos dados após retirada de determinados outliers')
    sem_outlier0 = Image.open('sem_outlier0.png')
    st.image(sem_outlier0, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier1 = Image.open('sem_outlier1.png')
    st.image(sem_outlier1, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier2 = Image.open('sem_outlier2.png')
    st.image(sem_outlier2, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier3 = Image.open('sem_outlier3.png')
    st.image(sem_outlier3, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier4 = Image.open('sem_outlier4.png')
    st.image(sem_outlier4, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier5 = Image.open('sem_outlier5.png')
    st.image(sem_outlier5, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    
    sem_outlier6 = Image.open('sem_outlier6.png')
    st.image(sem_outlier6, use_column_width=True)
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('Alguns valores destacados nos gráficos não foram considerados outliers a fim de não distanciar excessivamente os modelos da realidade.')
    
    
#Página 3
if pag == 'Sobre os modelos':
    st.title('Qualidade dos modelos')
    st.markdown('### Modelo para estimativa da saúde fiscal a partir do percentual do PIB comprometido para com a dívida pública')
    st.markdown('__Quantidade de variáveis independentes:__ 1')
    st.markdown('__R²:__ 59,67%')
    st.markdown('__Média residual:__ 0,033')
    st.markdown('__Desvio padrão residual:__ 0,088')
    st.markdown('__Erro quadrado médio:__ 0,009')
    st.markdwon(' ')
    st.markdwon(' ')
    
    dist_resid_fiscal_unemp = Image.open('dist_resid_fiscal_unemp.png')
    st.image(dist_resid_fiscal_unemp , use_column_width=True)  
    st.markdown('De acordo com o teste de Shapiro-Wilk, a distribuição dos resíduos é normal.')
    st.markdown('''
    Outro teste que pode confirmar a normalidade da distribuição dos resíduos e o teste de assimetria:
    
    __Fórmula Skewness:__
    
    Quanto mais próxima a zero, mais perfeita é a simetria, o que configura a normalidade. Para valores y > 0, existe uma assimetria positiva, e negativa para valores y < 0.
    __Resultado para Skewness:__ 0,808.
    
    A normalidade da distribuição não é perfeita, apresentando uma assimetria positiva.
    ''')
