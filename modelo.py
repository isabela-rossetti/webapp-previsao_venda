import streamlit as st
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'Calculo de propensão à compra de um produto', 
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-previsao-vendas')

@st.cache
def ler_dados():
	dados = pd.read_csv('dataset-vendas-tratado.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  


st.write('''
## Modelo criado para calcular a probabilidade de venda de um produto 

---
Para desenvolvimento do modelo foram usados dados fictícios uma empresa do setor financeiro que opera através de televendas para um de seus produtos.
Tais dados foram disponibilizados na a 10ª Competição de Machine Learning da [FLAI](https://www.flai.com.br/).


***Criado por [Isabela Rossetti](https://www.linkedin.com/in/isabelarossetti/)***. 

---

## Entre abaixo com as características do cliente e dos contatos feitos e verifique a provável resposta. 


''')

st.markdown('---') 
 
st.markdown('## Informações do cliente')
col1, col2 = st.columns(2)

x1 = col1.radio('Faixa de idade do cliente', dados['idade'].unique().tolist())
x2 = col2.radio('Tipo de ocupação do cliente', dados['trabalho'].unique().tolist())


st.markdown('---')

st.markdown('## Informações do(s) contato(s)')
col1, col2 = st.columns(2)

x3 = 'não'
x4 = col1.radio('Meio de contato com o cliente', dados['contato'].unique().tolist())
x5 = col1.radio('Último mês em que o contato foi feito', dados['mes'].unique().tolist())
x6 = col2.radio('Qual foi a duração do último contato?', dados['duracao'].unique().tolist()) 
x7 = col2.radio('Quantos contatos foram feitos com o cliente em campanhas anteriores?', dados['anterior'].unique().tolist()) 
 

dicionario  =  {'idade': [x1],
			'trabalho': [x2],
			'atraso': [x3],
			'contato': [x4],
			'mes': [x5],
			'duracao': [x6],
			'anterior': [x7]}

dados = pd.DataFrame(dicionario)  

st.markdown('---') 

st.markdown('## Executar o Modelo de Precificação') 

if st.button('CALCULAR A PROVÁVEL RESPOSTA DO CLIENTE'):
	st.markdown('---') 
	saida_label = predict_model(modelo, dados)['Label']
	saida_score = float(predict_model(modelo, dados)['Score']*100)
	st.markdown('### {},'.format(saida_label))
	st.markdown('### com probabilidade de **{:.1f}%** '.format(saida_score))
	st.markdown('---') 

