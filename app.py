import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

# 加载变量配置和模型
V = pd.read_excel("变量.xlsx")

# 指定的10个特征
cols = ['BMI', 'dietary_character', 'Vital_capacity', 'Pharyngeal_function',
        'Oral_function', 'Esophageal_function', 'Tongue_muscles',
        'Masticatory_and_buccal muscles', 'Pharyngeal_muscles', 'Shimmer']

# 加载多分类模型
model = {}
model["multi"] = joblib.load("Multi.pkl")

title = "Screening for Geriatric Dysphagia Patients (Multi-class)"

st.set_page_config(
    page_title=f"{title}",
    page_icon="logo.png",
    layout="wide"
)

T = "Screening for Geriatric Dysphagia Patients (Multi-class)"

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
        # 只显示指定的特征
        if i in cols:
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

# 确保数据框的列与特征列表一致
if len(pre.columns) != len(cols):
    # 只保留指定的特征列
    pre = pre[cols]

# 设置列名
pre.columns = cols

# 显示当前输入
with st.expander("Current inputs", True):
    st.dataframe(pre, use_container_width=True, hide_index=True)

# 预测按钮
button = st.button("Show screening results 🔍", use_container_width=True)

if not button:
    t = "Please set inputs and click button start predict!"
    st.markdown(f"<div style='{style}'>{t}</div>", unsafe_allow_html=True)

# 多分类预测逻辑
if button:
    try:
        # 进行预测
        r = model["multi"].predict_proba(pre)

        # 分类标签映射
        x1 = {"No Dysphagia": 0, "Oral Dysphagia": 1, "PharyngealDysphagia": 2, "Oropharyngeal Dysphagia": 3,
              "EsophagealDysphagia": 4}
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

        # 特征重要性可视化（修复缩进和模型引用）
        try:
            # 使用model["multi"]而不是model["binary"]
            final_estimator = model["multi"].steps[-1][1] if hasattr(model["multi"], 'steps') else model["multi"]
            feature_importances = final_estimator.feature_importances_
        except (AttributeError, IndexError):
            # 如果无法获取，则使用预设的列名列表
            feature_importances = [0] * len(cols)
            st.warning("无法获取特征重要性，请确保模型支持feature_importances_属性。")

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

    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")