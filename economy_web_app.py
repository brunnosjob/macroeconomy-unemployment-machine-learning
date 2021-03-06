#!/usr/bin/env python
# coding: utf-8

#Importando as bibliotecas
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
from PIL import Image
import pickle
    
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
st.sidebar.markdown("- [Artigo descrevendo o passo a passo do desenvolvimento do modelo de machine learning](https://br-cienciadedados.medium.com/projeto-de-machine-learning-iii-7960c6e0a6dc)")
st.sidebar.markdown(' ')

#Menu
st.sidebar.title('Menu')
pag = st.sidebar.selectbox('Selecione a página:', ['Interagir com o modelo', 'Sobre o conjunto de dados', 'Sobre o modelo e o produto final'])
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
    st.markdown(' ')    
    st.markdown(' ')

    #Explicações
    st.markdown('#### Como você pode experimentar e usar essa aplicação:')
    st.markdown('''
    
    1 - Você deve preencher os espaços, sendo dois deles opcionais.
    No espaço em que pede para inserir a saúde fiscal com valores percentuais, você deve preencher com um valor percentual de 0 a 100.
    Não se deve colocar o símbolo de porcentagem (%).
    O valor colocado por si só representa o percentual, isso é,
    a qualidade da saúde fiscal de um dado país (fictício ou real) em termos percentuais;

    2 - Você pode simplesmente inserir dados fictícios, dentro dos limites estabelecidos para a aplicação;

    3 - Você pode realizar uma pesquisa e usar dados reais:

    I - Escolha um país;
    
    II - Pesquise sobre a quantidade de pessoas desse país disponível ao mercado de trabalho, estando empregada ou não;

    III - Pesquise sobre a qualidade da saúde fiscal do país expressa em termos percentuais.
    O valor deve estar em porcentagem, quando for colocado no espaço de preenchimento, presente logo abaixo.

    III - Compare o resultado da predição acerca da taxa de desemprego com o valor real da taxa de desemprego
    desse país, assim como a quantidade estimada de desempregado, que o modelo apresenta, com a quantidade real de desempregado presente nesse país.
    Essa é uma forma de saber a qualidade do modelo.

    4 - Você pode repetir o processo inúmeras vezes, inclusive, usar essa aplicação web para algum trabalho, desde que a mesma caiba em seu trabalha,
    tendo em vista o que está ressaltado nas páginas __sobre o conjunto de dados__ e __sobre o modelo e o produto final__.''')
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ') 
    
    #Buscando modelo de previsão de taxa de desemprego
    with open('unemployment_simple_lm.pkl', 'rb') as f:
        regressor_sim_log = pickle.load(f)
        
    st.markdown('#### Test drive do modelo')

    #Aplicação dos modelos
    #Predição de saúde fiscal
    #Nome do país
    pais = st.text_input('Insira o nome do pais (Opcional):')
    
    #População trabalhadora
    worker_pop = st.number_input('''
    Insira a população total desse país apta para trabalhar.
    Esse valor é encontrado na Internet. É a soma do total empregado mais o total desempregado que busca emprego (Opcional):
    ''', 0, 8000000000, 0)

    #Valor percentual da dívida pública
    fiscal_health = st.number_input('Insira, em termos percentuais, a qualidade da saúde fiscal:', 0.0, 100.0, 0.01)

    #Predição de taxa de desemprego
    #Transformações e predição
    fiscal_health_0_1 = fiscal_health/100
    X_log = np.log(fiscal_health_0_1)
    X_log = X_log.reshape(-1,1)
    unemployment = regressor_sim_log.predict(X_log)
    unemployment = np.round(unemployment, 3)
    taxa = unemployment/100
    unemployed_pop = worker_pop * taxa

    st.write('País:', pais)
    st.write('A taxa de desemprego está em torno de {}%'.format(unemployment))
    if unemployed_pop > 0:
        st.write('O total de desempregado está em torno de {}.'.format(unemployed_pop))
    elif unemployed_pop == 0:
        st.write(' ')


    
#Página 2
if pag == 'Sobre o conjunto de dados':
    st.title('Qualidade dos dados')
    st.markdown('''
    Os dados são do primeiro semestre de 2021.
    Inicialmente, a base de dados continha 56 linhas, referentes a 56 países, e 44 colunas. As colunas apresentavam informações
    econômicas, de desempenho jurídico, de desempenho burocrático e de relação com a língua inglesa. Há países de todos os continentes e de diferentes desenvolvimento.
    Tais informações foram colhidas respectivamente da base de dados da organização Heritage e da EF English.
    
    Os dados não apresentavam ausência de valores tampouco inconsistência de valores, como idades negativas e/ou alturas negativas, informações irreais.
    
    Após a limpeza de dados e a análise de correlação, a base de dados para o desenvolvimento de modelos continha 44 linhas e 8 colunas.
    Diferentes modelos foram desenvolvidos até à seleção de dois.
    ''')
    st.markdown(' ')
    st.markdown(' ') 
    
    st.markdown('#### Correlações antes da retirada de outliers')
    st.markdown('Antes de a retirada de outliers, as principais correlações com a variável Unemployment (taxa de desemprego) eram:')
    corr_min = Image.open('corr_min.png')
    st.image(corr_min, use_column_width=True)
    st.markdown('''
    As principais correlações eram:
    
    #### Para com a variável dependente Unemployment (%):
    
    Inflation (%): 0,57;
    
    Fiscal Health: -0,58;
    
    GDP Growth Rate (%): -0,63.
    ''')
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ') 
    st.markdown('#### Correlações após retirada de outliers')
    
    
    st.markdown('Após a retirada de outliers, as principais correlações com a variável Unemployment (taxa de desemprego) eram:')
    sem_out_corr = Image.open('sem_out_corr.png')
    st.image(sem_out_corr, use_column_width=True)
    st.markdown('''
    As principais correlações eram:
    
    #### Para com a variável dependente Unemployment (%):
    
    Public Debt (% of GDP): 0,51
    
    Fiscal Health: -0,68;
    ''')
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ') 
    st.markdown('''Como pode-se apreender, houve potencialização na correlação entre certas variáveis independentes e a variável dependente, enquanto, 
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
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('#### Fonte de dados')
    st.markdown('''
    EF ENGLISH. English Proficiency Index. 2021. Página Download.
    Disponível em : https://www.ef.com/wwen/epi/. Acesso em abril 2021.
    
    HERITAGE. 2021 Index of Economic Freedom. 2021. Página explore the data.
    Disponível em: https://www.heritage.org/index/ . Acesso em abril 2021.
    ''')
    
#Página 3
if pag == 'Sobre o modelo e o produto final':
    st.title('Qualidade dos modelos')
    st.markdown('__Antes de tudo... Importância de um modelo como esse__')
    st.markdown('''
    Primeiramente, destaco que um modelo como esse, bem desenvolvido, pode auxiliar na tomada de decisão do Governo acerca 
    da saúde fiscal, tendo em vista o possível impacto que essa variável pode advir negativamente sobre a taxa de desemprego.
    Além de questões de administração pública, um modelo como esse pode auxiliar em trabalhos acadêmicos, como a dissertação de um artigo científico,
    o qual é útil para fins políticos e jornalísticos.
    ''')
    
    

    st.markdown('### Modelo para estimativa da taxa de desemprego a partir do percentual que representa a saúde fiscal')
    
    st.markdown('''
    Inicio apresentando a comparação entre a linha de predição e a linha real. 
    A linha de predição, em laranja, 
    demonstra o desempenho do  modelo de predição de saúde fiscal, em relação aos dados de teste, os quais são representados pela linha azul.
    As previsões são razoáveis em relação aos dados reais.''')
    pred_fiscal_unemp = Image.open('pred_fiscal_unemp.png')
    st.image(pred_fiscal_unemp , use_column_width=True)
    
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ')  
    
    st.markdown('#### Estatísticas do desempenho do modelo em relação aos dados de teste')
    
    st.markdown('__Quantidade de variáveis independentes:__ 1')
    st.markdown('__R²:__ 45.23%')
    st.markdown('__Média residual:__ -1.526')
    st.markdown('__Desvio padrão residual:__ 2.014')
    st.markdown('__Erro quadrado médio:__ 6.387')
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ') 

    st.markdown('#### Distribuição do resíduo')
    
    dist_resid_model_2 = Image.open('dist_resid_fiscal_unemp.png')
    st.image(dist_resid_model_2 , use_column_width=True)  
    st.markdown('De acordo com o teste de Shapiro-Wilk, a distribuição dos resíduos é normal. Observando o gráfico, pode-se notar essa evidência.')
    st.markdown('''
    Outro teste que confirma ou rejeita a normalidade é o teste de Jarque-Bera:
    
    __Jarque-Bera:__
    
    Para valores de p < 0,05 a normalidade é rejeitada.
    
    __Resultado de P de Jarque-Bera:__ 0.602
    
    Confirmada novamente a normalidade dos resíduos.
    
    Uma configuração que pode confirmar a normalidade da distribuição dos resíduos é a análise da assimetria:
    
    __Fórmula Skewness:__
    
    Quanto mais próxima a zero, mais perfeita é a simetria, o que configura a normalidade. Para valor y > 0, existe uma assimetria positiva, e negativa para valor y < 0.
    
    __Resultado para Skewness:__ 0,495.
    
    A normalidade da distribuição não é perfeita, apresentando uma assimetria positiva.
    ''')
    
    st.markdown('''
    Um último valor que evidencia mais informações é o valor da curtose.
    
    __Fórmula Curtose:__
    
    A curtose de uma distribuição normal é 3. Para valor y > 3 a distribuição é mais “alta” que a distribuição normal e para valor y < 3, mais “achatada”.
    
    __Resultado para Curtose:__ -0,871
    
    A curva normal do resíduo do modelo é mais achatada do que a curva da perfeita curva normal.
    ''')
    
    st.markdown(' ')
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('O gráfico a seguir evidencia a suposta dependência dos resíduos, porquanto, o gráfico não apresenta uma distribuição uniforme em torno de zero.')
    
    scatter_resid_unemp = Image.open('222.png')
    st.image(scatter_resid_unemp , use_column_width=True)
    
    st.markdown('''
    __Testando a independência dos resíduos__
    
    __Resultado para independência dos resíduos:__ Durbin-Watson = 0,94333
    
    Uma regra geral seguida é que os valores estatísticos do teste Durbin-Watson na faixa de 1,5 a 2,5 são relativamente aceitáveis. 
    Valores fora desse intervalo podem ser motivos de preocupação. Valores abaixo de 1 ou acima de 3 são um motivos definitivos de preocupação.
    O valor retornado é aproximadamente 0,943. Esse é um valor inferior a 1, portanto, é um valor definitivamente preocupante, evidenciando correlação positiva entre os resíduos.
    O teste evidencia que não há independência dos resíduos.
    ''')
    
    st.markdown('''
    __Resultado para homocedasticidade:__ p de Goldfeld-Quandt: 0.14765
    
    Se p < 0,05, não há homocedasticidade;
    
    Se p >= 0,05, há homocedasticidade;
    
    Portanto, há homocedasticidade.
    ''')
    
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ')  
    
    st.markdown('Por fim, apresento o gráfico de lineariedade entre a variável Public Debt (% of GDP) e Fiscal Health.')
    
    scttr_unemp_fiscal = Image.open('scatter_unemp_fiscal.png')
    st.image(scttr_unemp_fiscal, use_column_width=True)
    
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ') 
    
    st.markdown('#### Conclusão acerca do modelo preditivo para Fiscal Health (saúde fiscal)')
    st.markdown('''
    O modelo gera resíduos que cumprem com a maioria das premissas de um bom modelo linear:
                
     1 - Os resíduos apresentam distribuição normal;
                
     2 - O erro quadrado médio é o menor dentre os modelos desenvolvidos;
                
     3 - Os resíduos não são independentes;
     
     4 - A variância dos resíduos é constante para cada valor condicional de X, ou seja, há homocedasticidade.
                
     Esse modelo é um estimador com qualidade moderada, não cumprindo com a premissa de independência dos resíduos. 
     As premissas de linearidade, normalidade da distribuição dos erros e de homocedasticidade são cumpridas.''')
    
    st.markdown(' ')
    st.markdown(' ')    
    st.markdown(' ')
    st.markdown(' ') 

    st.markdown('### Alerta sobre o desempenho do modelo')
    st.markdown('''
    __O modelo é simplório para abarcar a complexidade econômica__, 
    porém, apresentou consideráveis aproximações aos dados reais em diferentes testes. 
    
    __O produto final é uma demonstração__ de um produto de machine learning voltado para as Ciências Econômicas. 
    Portanto, __esse é um produto de portfólio, e não cumpre com o rigor que um modelo econométrico deve de fato ter__, 
    ainda que tenha sido razoável e tenha sido bom em diferentes testes e cenários preditivos.
    ''')
