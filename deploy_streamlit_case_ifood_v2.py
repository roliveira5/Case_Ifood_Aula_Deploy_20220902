import streamlit as st
import pandas as pd
import numpy as np
import pickle

#streamlit
def main():        
    
    st.set_page_config(page_title = 'Simulador - Case Ifood',\
                       page_icon = 'logo_dh.jpeg',
                       layout='wide',
                       initial_sidebar_state = 'expanded')
    
    
    st.title('Simulador - Conversão de Vendas')
    with st.expander('Descrição do App',expanded=False):
        st.markdown('O objetivo principal desta ferramenta é realizar predições sobre a chance de um cliente converter em uma dada campanha de mkt...')
    
#################################################################################################################
    with st.sidebar:
        database = st.radio('fonte dos dados de entrada (X):',('Manual', 'CSV'))
        
        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Selecione o arquivo CSV contendo as colunas acima descritas',type='csv')
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lgbm_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)
        else:
            X1 = st.slider('Total gasto em carnes nos ult. 2anos (USD)',0,2000,step=50)
            X2 = st.slider('Recencia desde a última compra (dias)',0,100,step=1)
            X3 = st.slider('Tempo como Cliente (anos)',7,10,step=1)
            X4 = st.slider('Renda familiar anual (USD)',2000,163000,step=500)
            X5 = st.slider('Total gasto em vinhos nos ult. 2anos (USD)',0,2000,step=50)
            X6 = st.slider('Total gasto em prod. Gold nos ult. 2anos (USD)',0,300,step=10)
            X7 = st.slider('Idade (anos)',18,120,step=1)
            X8 = st.slider('Total gasto em doces nos ult. 2anos (USD)',0,2000,step=50)
            X9 = st.slider('Total gasto em frutas nos ult. 2anos (USD)',0,300,step=10)
            X10 = st.slider('Total gasto em peixes nos ult. 2anos (USD)',0,300,step=10)
            X11 = st.slider('Num de compras realizadas na loja',0,30,step=1)
            X12 = st.slider('Num de visitas mensais ao site',0,30,step=1)
            X13 = st.slider('Num de compras atraves do catalogo',0,30,step=1)
            X14 = st.slider('Num de compras pelo site',0,30,step=1)
            X15 = st.slider('Num de compras com desconto',0,30,step=1)

            Xtest = pd.DataFrame({'MntMeatProduct': [X1], 'Recency': [X2], 'Time_Customer': [X3], 'Income': [X4], 
                                      'MntWines': [X5], 'MntGoldProds': [X6], 'Age': [X7], 'MntSweetProducts': [X8], 
                                      'MntFruits': [X9], 'MntFishProducts': [X10], 'NumStorePurchases': [X11], 
                                      'NumWebVisitsMonth': [X12], 'NumCatalogPurchases': [X13], 
                                      'NumWebPurchases': [X14], 'NumDealsPurchases': [X15]})
            
            mdl_lgbm = pickle.load(open('pickle_mdl_lgbm_select.pkl', 'rb'))
            ypred = mdl_lgbm.predict(Xtest)
                                     
##################################################################################################################

    if database == 'Manual':
        with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
        with st.expander('Visualizar Predição', expanded = False):
                if ypred==0:
                    c1, c2 = st.columns([1,2])
                    c1.error(ypred[0])
                else:
                    c1, c2 = st.columns([1,2])
                    c1.success(ypred[0])

        df_download = Xtest.copy()
        df_download['Response_pred'] = ypred
        #st.dataframe(df_download)
        csv = df_download.to_csv(sep=',',decimal=',',index=False)     
        st.download_button(label="Download CSV ",
                                data = csv,
                                file_name = "Prediction.csv",
                                mime = 'text/csv')
            

    else: #database == 'CSV'
        if file:
            with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
            with st.expander('Visualizar Predições', expanded = False):
                st.dataframe(ypred)            

            df_download = Xtest.copy()
            df_download['Response_pred'] = ypred
            st.write(df_download.shape)
            #st.dataframe(df_download)
            csv = df_download.to_csv(sep=',',decimal=',',index=False)
            st.download_button(label="Download CSV",
                                data = csv,
                                file_name = "Predictions.csv",
                                mime = 'text/csv')

if __name__ == '__main__':
	main()