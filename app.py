import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

V = pd.read_excel("变量.xlsx")
cols = ['sex', 'AGE', 'BMI', 'tooth_loss', 'TV', 'sitting_position_change', 'Eating_position', 'dietary_character', 'appetite', 'cerebral_hemorrhage', 'Alzheimers_disease', 'Parkinsons', 'multiple_sclerosis', 'sarcopenia', 'coronary_heart_disease', 'traumatic_brain_injury', 'Dysfunction_of_dental_chewing', 'Vital_capacity', 'Respiratory_and_swallowing_coordination', 'History_of pneumonia', 'Nutritional_status', 'Pharyngeal_function', 'Oral_function', 'Esophageal_function', 'Airway_protection_function', 'Tongue_muscles', 'Masticatory_and_buccal muscles', 'Pharyngeal_muscles', 'MPT', 'F0Hz', 'Jitter', 'Shimmer', 'HNR', 'Grade', 'Roughness', 'Breathiness', 'Asthenia', 'Strain']

model = {}
model["multi"] = joblib.load("Multi.pkl")
model["binary"] = joblib.load("Binary.pkl")

#data = pd.read_csv("data.csv")
#data = data.iloc[:, :]
#data = data.dropna(axis=0, how='any')
#data = data.drop('stage',axis=1)

title = "Screening for Geriatric Dysphagia Patients (Multi-class & binary)"

st.set_page_config(    
    page_title=f"{title}",
    page_icon="logo.png",
    layout="wide"
)

M = ["Multi-class Model", "Binary Model"]
M1 = ["multi", "binary"]
T = ["Screening for Geriatric Dysphagia Patients (Multi-class)",
     "Screening for Geriatric Dysphagia Patients (binary)"]

header_style = """
    text-align: center;
    font-size: 28px;
    border-bottom: 1px solid black;
    margin-bottom: 15px;
"""

style = """
    text-align: center;
    font-size: 20px;
    border-bottom: 1px solid black;
    margin-bottom: 15px;
    color: white; 
    background: red; 
    padding: 0.5rem; 
    border: 1px solid red; 
    border-radius: 0.5rem;
"""

st.sidebar.image("logo1.png", use_container_width=True)

st.sidebar.markdown(f"<div style='{header_style}'>Model select</div>", unsafe_allow_html=True)     
m = st.sidebar.selectbox("Select Prediction Model Type", M, label_visibility="collapsed")

st.sidebar.markdown(f"<div style='{header_style}'>Model variable inputs</div>", unsafe_allow_html=True)
predata = {}
with st.sidebar:
    for i, j, k in zip(V["变量名称"].tolist(), V["变量描述"].tolist(), V["取值"].tolist()):
        if "step" in str(k):
            k = eval(k)
            if j.startswith("("):
                predata[i] = st.number_input(i+j, min_value=k["min"]+k["step"]-k["step"], max_value=k["max"]+k["step"]-k["step"], step=k["step"])
            elif str(j)!="nan":
                predata[i] = st.number_input(i, min_value=k["min"]+k["step"]-k["step"], max_value=k["max"]+k["step"]-k["step"], step=k["step"], help=j)
            else:
                predata[i] = st.number_input(i, min_value=k["min"]+k["step"]-k["step"], max_value=k["max"]+k["step"]-k["step"], step=k["step"])
        else:
            k = eval(k)
            if str(j)!="nan":
                predata[i] = k[st.selectbox(i, k, help=j)]
            else:
                predata[i] = k[st.selectbox(i, k)]

pre = pd.DataFrame([predata])
pre.columns = cols
    
with st.expander("Current inputs", True):
    st.dataframe(pre, use_container_width=True, hide_index=True)

button = st.button("Show screening results 🔍", use_container_width=True)

if not button:
    t = "Please set inputs and click button start predict!"
    st.markdown(f"<div style='{style}'>{t}</div>", unsafe_allow_html=True)
    
if m==M[0] and button:
    r = model[M1[0]].predict_proba(pre)
    x1 = {"No Dysphagia":0,"Oral Dysphagia":1,"PharyngealDysphagia":2,"Oropharyngeal Dysphagia":3,"EsophagealDysphagia":4}
    x2 = {j:i for i, j in list(x1.items())}
    label = list(x2.values())
    
    st.markdown(f"<h1 style='{header_style}'>{T[0]}</h1>", unsafe_allow_html=True)
    
    d = {"class":label, "percent":r.tolist()[0], "color":label, "text":[round(i, 3) for i in r.tolist()[0]]}
     
    fig = px.bar(d, x='class', y='percent', color="color", text="text")
    fig.update_traces(textposition='outside')  
    fig.update_layout(  
        title={'text': 'Predictions', 'x': 0.5, 'xanchor': 'center'}, 
        legend={'x': 0.5, 'xanchor': 'center', 'y': 1.1, 'orientation':'h', 'title':''}  
    )
    #fig.update_yaxes(range=[0, 1])
    
    with st.expander("", True):
        st.plotly_chart(fig, use_container_width=True)
    
    d1 = pd.DataFrame({
        "feature-name":model[M1[0]].feature_names_in_, 
        "feature-importance":model[M1[0]].feature_importances_, 
        "text": [str(round(i, 3)) for i in model[M1[0]].feature_importances_]
    })
    d1.sort_values(by="feature-importance", ascending=False, inplace=True)
    
    fig = px.bar(d1, x='feature-name', y='feature-importance', color="feature-importance", text="text")
    fig.update_traces(  
        textposition='outside',  
        textfont=dict(size=14),
        textangle=0  
    )  
    fig.update_layout(  
        title={'text': 'Feature importance', 'x': 0.5, 'xanchor': 'center'}, 
        showlegend=False,
    )
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(range=[0, 0.225])
    
    with st.expander("", True):
        st.plotly_chart(fig, use_container_width=True)
elif button:
    r = model[M1[1]].predict_proba(pre)
    
    x1 = {"No Dysphagia":0,"Dysphagia":1}
    x2 = {j:i for i, j in list(x1.items())}
    label = label = list(x2.values())
    
    st.markdown(f"<h1 style='{header_style}'>{T[1]}</h1>", unsafe_allow_html=True)
    
    d = {"class":label, "percent":r.tolist()[0], "color":label, "text":[round(i, 3) for i in r.tolist()[0]]}
     
    fig = px.bar(d, x='class', y='percent', color="color", text="text")
    fig.update_traces(textposition='outside')  
    fig.update_layout(  
        title={'text': 'Predictions', 'x': 0.5, 'xanchor': 'center'}, 
        legend={'x': 0.5, 'xanchor': 'center', 'y': 1.1, 'orientation':'h', 'title':''}  
    )
    #fig.update_yaxes(range=[0, 1])
    
    with st.expander("", True):
        st.plotly_chart(fig, use_container_width=True)
    
    d1 = pd.DataFrame({
        "feature-name":model[M1[1]].feature_names_in_, 
        "feature-importance":model[M1[1]].feature_importances_, 
        "text": [str(round(i, 3)) for i in model[M1[1]].feature_importances_]
    })
    d1.sort_values(by="feature-importance", ascending=False, inplace=True)
    
    fig = px.bar(d1, x='feature-name', y='feature-importance', color="feature-importance", text="text")
    fig.update_traces(  
        textposition='outside',  
        textfont=dict(size=14),
        textangle=0  
    )  
    fig.update_layout(  
        title={'text': 'Feature importance', 'x': 0.5, 'xanchor': 'center'}, 
        showlegend=False,
    )
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(range=[0, 0.225])
    
    with st.expander("", True):
        st.plotly_chart(fig, use_container_width=True)

