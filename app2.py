import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

# 加载变量配置和模型
V = pd.read_excel("变量副本.xlsx")
cols = ['dietary_character', 'Vital_capacity', 'Pharyngeal_function', 'Oral_function', 'Esophageal_function', 'Airway_protection_function', 'Masticatory_and_buccal muscles', 'F0Hz', 'Jitter', 'Shimmer']

# 加载二分类模型
model = {}
model["binary"] = joblib.load("Binary.pkl")

# 设置页面配置
title = "Screening for Geriatric Dysphagia Patients (Binary)"
st.set_page_config(
    page_title=f"{title}",
    page_icon="logo.png",
    layout="wide"
)

# 页面标题和样式
T = "Screening for Geriatric Dysphagia Patients (Binary Classification)"

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

# 侧边栏布局
st.sidebar.image("logo1.png", use_container_width=True)
st.sidebar.markdown(f"<div style='{header_style}'>Model variable inputs</div>", unsafe_allow_html=True)

# 收集用户输入
predata = {}
with st.sidebar:
    for i, j, k in zip(V["变量名称"].tolist(), V["变量描述"].tolist(), V["取值"].tolist()):
        if "step" in str(k):
            k = eval(k)
            if j.startswith("("):
                predata[i] = st.number_input(i + j, min_value=k["min"] + k["step"] - k["step"],
                                             max_value=k["max"] + k["step"] - k["step"], step=k["step"])
            elif str(j) != "nan":
                predata[i] = st.number_input(i, min_value=k["min"] + k["step"] - k["step"],
                                             max_value=k["max"] + k["step"] - k["step"], step=k["step"], help=j)
            else:
                predata[i] = st.number_input(i, min_value=k["min"] + k["step"] - k["step"],
                                             max_value=k["max"] + k["step"] - k["step"], step=k["step"])
        else:
            k = eval(k)
            if str(j) != "nan":
                predata[i] = k[st.selectbox(i, k, help=j)]
            else:
                predata[i] = k[st.selectbox(i, k)]

# 转换输入为DataFrame
pre = pd.DataFrame([predata])
pre.columns = cols

# 显示当前输入
with st.expander("Current inputs", True):
    st.dataframe(pre, use_container_width=True, hide_index=True)

# 预测按钮
button = st.button("Show screening results 🔍", use_container_width=True)

if not button:
    t = "Please set inputs and click button start predict!"
    st.markdown(f"<div style='{style}'>{t}</div>", unsafe_allow_html=True)

# 二分类预测逻辑
if button:
    r = model["binary"].predict_proba(pre)

    # 分类标签映射
    x1 = {"No Dysphagia": 0, "Dysphagia": 1}
    x2 = {j: i for i, j in list(x1.items())}
    label = list(x2.values())

    # 页面标题
    st.markdown(f"<h1 style='{header_style}'>{T}</h1>", unsafe_allow_html=True)

    # 预测结果可视化
    d = {"class": label, "percent": r.tolist()[0], "color": label, "text": [round(i, 3) for i in r.tolist()[0]]}

    fig = px.bar(d, x='class', y='percent', color="color", text="text")
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title={'text': 'Predictions', 'x': 0.5, 'xanchor': 'center'},
        legend={'x': 0.5, 'xanchor': 'center', 'y': 1.1, 'orientation': 'h', 'title': ''}
    )

    with st.expander("Prediction Results", True):
        st.plotly_chart(fig, use_container_width=True)

        # 特征重要性可视化
    try:
        # 动态获取Pipeline的最后一个步骤
        final_estimator = model["binary"].steps[-1][1]
        feature_importances = final_estimator.feature_importances_
    except (AttributeError, IndexError):
        # 如果无法获取，则使用预设的列名列表
        feature_importances = [0] * len(cols)
        st.warning("无法获取特征重要性，请确保Pipeline中的最后一步是决策树、随机森林等支持特征重要性的模型。")

    d1 = pd.DataFrame({
        "feature-name": cols,
        "feature-importance": feature_importances,
        "text": [str(round(i, 3)) for i in feature_importances]
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

    with st.expander("Feature Importance", True):
        st.plotly_chart(fig, use_container_width=True)