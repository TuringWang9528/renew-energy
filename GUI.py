import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- 1. 加载模型 ---
# 确保你的模型文件 'Xgboost.pkl' 和这个脚本在同一个目录下
try:
    model = joblib.load('Xgboost.pkl')
except FileNotFoundError:
    st.error("模型文件 'Xgboost.pkl' 未找到，请确保它和脚本在同一目录。")
    st.stop()


# --- 2. Streamlit 用户界面 ---
st.title("光伏项目10年净现值 (NPV) 预测平台")
st.write("请在下方输入光伏项目的相关参数，然后点击“预测”按钮。")

# 定义特征名称，需与模型训练时完全一致
feature_names = [
    'solar_irradiance_kWh_m2_day', 'system_size_kW', 'panel_efficiency_pct',
    'inverter_efficiency_pct', 'install_cost_CNY_per_kW', 'o&m_CNY_per_kW_year',
    'electricity_price_CNY_per_kWh', 'subsidy_CNY_per_kWh', 'discount_rate_pct'
]

# --- 3. 创建与你的数据集匹配的输入框 ---
st.header("项目参数输入")

# 使用列布局让界面更美观
col1, col2, col3 = st.columns(3)

with col1:
    solar_irradiance = st.number_input("日太阳辐照度 (kWh/m²/day)", min_value=3.0, max_value=6.0, value=4.7, format="%.3f")
    system_size = st.number_input("系统装机容量 (kW)", min_value=3.0, max_value=7.0, value=5.0, format="%.3f")
    panel_efficiency = st.number_input("光伏板效率 (%)", min_value=15.0, max_value=25.0, value=19.5, format="%.2f")

with col2:
    inverter_efficiency = st.number_input("逆变器效率 (%)", min_value=94.0, max_value=99.0, value=96.5, format="%.2f")
    install_cost = st.number_input("单位安装成本 (元/kW)", min_value=4000, max_value=9000, value=6500)
    o_and_m_cost = st.number_input("年运维成本 (元/kW/年)", min_value=50, max_value=200, value=120)

with col3:
    electricity_price = st.number_input("上网电价 (元/kWh)", min_value=0.4, max_value=1.0, value=0.7, format="%.3f")
    subsidy = st.number_input("度电补贴 (元/kWh)", min_value=0.0, max_value=0.2, value=0.1, format="%.3f")
    discount_rate = st.number_input("折现率 (%)", min_value=3.0, max_value=10.0, value=6.0, format="%.2f")


# --- 4. 处理输入并进行预测 ---
# 将所有输入控件的值收集到一个列表中
feature_values = [
    solar_irradiance, system_size, panel_efficiency, inverter_efficiency,
    install_cost, o_and_m_cost, electricity_price, subsidy, discount_rate
]

# 将特征列表转换为Numpy数组，并 reshape 成模型需要的格式 (1, 9)
features = np.array([feature_values])
features_df = pd.DataFrame(features, columns=feature_names)


if st.button("预测 NPV"):
    # 使用 .predict() 进行预测
    prediction = model.predict(features)[0]

    st.success(f"**预测的10年净现值 (NPV):  ¥ {prediction:,.2f}**")

    st.write("---")

    # --- 5. SHAP 可解释性分析 ---
    st.header("模型预测解释")
    st.write("下图展示了各个输入特征如何影响本次的预测结果。红色特征推动预测值升高，蓝色特征推动预测值降低。")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    # 生成 HTML 版本的 SHAP 图
    # shap.force_plot 默认返回一个可以渲染为 HTML 的对象
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=features_df.iloc[0]
    )

    # 将绘图对象转换为 HTML 字符串
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    # 使用 st.components.v1.html 显示 SHAP 图
    components.html(shap_html, height=200)
        st.pyplot(fig, use_container_width=True)
