import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- 页面基础设置 ---
st.set_page_config(
    page_title="光伏项目NPV预测平台",
    page_icon="☀️",
    layout="wide"
)

# --- 辅助函数：将DataFrame转换为CSV，用于下载 ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 1. 加载模型 ---
# 将模型加载放在函数中并用缓存，避免每次交互都重新加载
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Xgboost.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("模型文件 'Xgboost.pkl' 未找到，请确保它和脚本在同一目录。")
    st.stop()

# 定义特征名称，需与模型训练时完全一致
feature_names = [
    'solar_irradiance_kWh_m2_day', 'system_size_kW', 'panel_efficiency_pct',
    'inverter_efficiency_pct', 'install_cost_CNY_per_kW', 'o&m_CNY_per_kW_year',
    'electricity_price_CNY_per_kWh', 'subsidy_CNY_per_kWh', 'discount_rate_pct'
]


# --- 2. 侧边栏导航 ---
st.sidebar.title("导航")
app_mode = st.sidebar.radio(
    "请选择功能模式",
    ["实时单次预测", "批量文件预测"]
)


# =====================================================================================
# --- 模式一：实时单次预测 (原功能) ---
# =====================================================================================
if app_mode == "实时单次预测":
    st.title("☀️ 光伏项目10年净现值 (NPV) 实时预测")
    st.write("请在下方输入光伏项目的相关参数，然后点击“预测”按钮。")

    st.header("项目参数输入")
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

    feature_values = [
        solar_irradiance, system_size, panel_efficiency, inverter_efficiency,
        install_cost, o_and_m_cost, electricity_price, subsidy, discount_rate
    ]
    features_df = pd.DataFrame([feature_values], columns=feature_names)

    if st.button("预测 NPV", type="primary"):
        prediction = model.predict(features_df)[0]
        st.success(f"**预测的10年净现值 (NPV):  ¥ {prediction:,.2f}**")
        st.write("---")

        st.header("本次预测解释")
        st.write("下图展示了各个输入特征如何影响本次的预测结果。红色特征推动预测值升高，蓝色特征推动预测值降低。")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0], features_df.iloc[0])
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(shap_html, height=200)

# =====================================================================================
# --- 新功能：批量文件预测 ---
# =====================================================================================
elif app_mode == "批量文件预测":
    st.title("📂 批量文件预测")
    st.write("请上传CSV文件进行批量预测。文件需要包含与单次预测相同的9个特征列，顺序也需一致。")

    uploaded_file = st.file_uploader("选择一个CSV文件", type="csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            # 验证上传文件的列名是否正确
            if list(input_df.columns) == feature_names:
                st.info("文件上传成功，正在进行预测...")
                predictions = model.predict(input_df)
                result_df = input_df.copy()
                result_df['Predicted_NPV_10yr_CNY'] = predictions

                st.dataframe(result_df)

                csv_to_download = convert_df_to_csv(result_df)
                st.download_button(
                   label="下载预测结果",
                   data=csv_to_download,
                   file_name='predicted_results.csv',
                   mime='text/csv',
                )
            else:
                st.error("文件列名或顺序不正确。请确保文件包含以下列，且顺序一致：")
                st.json(feature_names)

        except Exception as e:
            st.error(f"处理文件时出错: {e}")


# =====================================================================================
# --- 新功能：模型总体解读 ---
# =====================================================================================
st.write("---")
with st.expander("🔬 查看模型总体特征重要性"):
    st.write("""
    下图（SHAP摘要图）展示了模型在全局范围内的行为。它告诉我们哪些特征对所有预测的平均影响最大。
    - **特征重要性**: Y轴上的特征按重要性从上到下排序。
    - **影响方向**: 点的颜色表示特征值的大小（红色为高，蓝色为低）。
    - **分布**: X轴表示该特征对预测结果的影响（SHAP值）。例如，一个靠右的红点意味着该特征的一个高取值会推高最终的预测NPV。
    """)
    
    # 为了生成摘要图，我们需要一部分数据作为背景参考。
    # 这里我们使用您最初提供的数据作为示例样本。在实际应用中，通常会使用部分训练集或验证集。
    sample_data = {
        'solar_irradiance_kWh_m2_day': [5.048, 4.378, 5.217, 4.895, 3.688, 5.451, 5.022, 5.072, 3.756, 4.401],
        'system_size_kW': [5.473, 5.773, 5.842, 5.007, 5.041, 5.6, 4.629, 5.675, 4.988, 4.232],
        'panel_efficiency_pct': [16.5, 19.67, 17.03, 17.22, 21.06, 19.14, 23.37, 18.55, 21.81, 19.69],
        'inverter_efficiency_pct': [97.36, 98.8, 95.85, 96.13, 98.06, 96.53, 97.68, 96.3, 95.08, 97.73],
        'install_cost_CNY_per_kW': [8292, 6509, 8788, 7429, 5022, 8483, 6702, 5137, 5187, 7594],
        'o&m_CNY_per_kW_year': [55, 152, 193, 174, 92, 58, 191, 191, 130, 146],
        'electricity_price_CNY_per_kWh': [0.823, 0.717, 0.56, 0.786, 0.488, 0.798, 0.715, 0.5, 0.486, 0.788],
        'subsidy_CNY_per_kWh': [0.083, 0.114, 0, 0.129, 0.144, 0.113, 0.099, 0.162, 0.136, 0.056],
        'discount_rate_pct': [8.01, 5.5, 8.96, 4.69, 6.77, 4.29, 6.47, 4.07, 4.13, 7.29]
    }
    sample_df = pd.DataFrame(sample_data)

    # --- START: 关键修复代码 ---
    # 检查并修正模型内部的 base_score 参数格式
    try:
        booster = model.get_booster()
        # 获取原始配置
        config = booster.save_config()
        # 清理 base_score 字符串 (去除方括号)
        config = config.replace('"base_score":"[', '"base_score":"').replace(']"', '"')
        # 将修正后的配置加载回模型
        booster.load_config(config)
    except Exception as e:
        st.warning(f"修正模型配置时出现轻微错误: {e}. 仍将尝试继续...")
    # --- END: 关键修复代码 ---

    explainer_global = shap.TreeExplainer(model)
    shap_values_global = explainer_global.shap_values(sample_df)

    # 创建一个新图形，避免 Streamlit 的缓存警告
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values_global, sample_df, show=False, plot_size=None)
    plt.tight_layout() # 调整布局，防止标签重叠
    st.pyplot(fig)
