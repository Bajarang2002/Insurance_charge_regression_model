import streamlit as st
import pickle
import numpy as np

lr=pickle.load(open("Linear_Regressor.pkl","rb"))
dt=pickle.load(open("Decision_Tree.pkl","rb"))
rf=pickle.load(open("Random_Forest.pkl",'rb'))

st.title('Insurance Charge Prediction Web app')
st.subheader('Fill the detail to predict insurance charges')

model=st.sidebar.selectbox('Choose the ML model',['Lin_reg','DT_reg','RF_reg'])

age=st.slider(' Age',18,65,1)
sex=st.selectbox('Sex',['Male','Female'])
bmi=st.slider('BMI',10,60,1)
children=st.selectbox('Children',[0,1,2,3,4,5])
smoker=st.selectbox('Smoker',['Yes','No'])
region=st.selectbox('Region',['northwest','northeast','southwest','southeast'])

if st.button('Predict Insurance Charges'):
    if sex=='Male':
        sex=1
    else:
        sex=0
    if smoker=='Yes':
        smoker= 1
    else:
        smoker =0
    if region=='northwest':
        nwest=1
        neast=0
        swest=0
        seast=0
    elif region=='southwest':
        nwest=0
        neast=0
        swest=1
        seast=0
    elif region=='southeast':
        nwest=0
        neast=0
        swest=0
        seast=1
    else:
        nwest=0
        neast=1
        swest=0
        seast=0
    
test =np.array([age,sex,bmi,children,smoker,nwest,swest,seast]) 
test=test.reshape(1,8)
if model=="Lin_reg":
    st.success(lr.predict (test)[0])
elif model=="DT_reg":
    st.success(dt.predict(test)[0])
else:
    st.success(rf.predict(test)[0])


