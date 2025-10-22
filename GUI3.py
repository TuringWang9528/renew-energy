# GUI2.py
# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ========= 关键兼容补丁：必须在导入 shap 之前 =========
# NumPy 2.x 移除了 np.int/np.float 等别名；老版本 SHAP 仍可能调用它们
if not hasattr(np, "int"):
    np.int = int  # 临时补回；根因还是版本不匹配，但这样可保证先跑通

import shap  # 在补丁之后再导入

# --- 页面基础设置 ---
st.set_page_config(
    page_title="光伏项目NPV预测平台",
    page_icon="☀️",
    layout="wide"
)

# --- 辅助函数：将DataFrame转换为CSV，用于下载 ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# --- 1. 加载模型 ---
@st.cache_resource
def load_model():
    import xgboost as xgb
    import joblib
    try:
        mdl = joblib.load('Xgboost.pkl')  # 先按原方式读进来（可能是 2.x 训练的）
        # 将旧包装器里的 Booster 导出为 JSON，再用当前版本重建
        try:
            booster = mdl.get_booster()
            booster.save_model('Xgboost.json')
            reg = xgb.XGBRegressor()
            reg.load_model('Xgboost.json')   # 用当前 xgboost 版本加载
            return reg
        except Exception:
            # 没有 get_booster（或其它问题）就直接返回原模型，让外层报错信息可见
            return mdl
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"加载模型时出错：{e}")
        return None

model = load_model()
if model is None:
    st.error("模型文件 'Xgboost.pkl' 未找到或无法加载，请确保它与脚本在同一目录且可读。")
    st.stop()

# 定义特征名称，需与模型训练时完全一致
feature_names = [
    'solar_irradiance_kWh_m2_day', 'system_size_kW', 'panel_efficiency_pct',
    'inverter_efficiency_pct', 'install_cost_CNY_per_kW', 'o&m_CNY_per_kW_year',
    'electricity_price_CNY_per_kWh', 'subsidy_CNY_per_kWh', 'discount_rate_pct'
]

# --- 构建 SHAP 解释器（带回退） ---
def _make_explainer(model, background_df: pd.DataFrame):
    """
    优先使用 TreeExplainer；如果遇到 XGBoost/SHAP 版本兼容问题，回退到通用 Explainer。
    统一使用新版调用：explainer(X) -> Explanation
    """
    try:
        return shap.TreeExplainer(model)
    except Exception:
        # 回退到通用 Explainer（可能走 Kernel/Linear/Partition，速度略慢但稳定）
        return shap.Explainer(model, background_df)

@st.cache_resource
def get_explainer_for_background(background_df_hash: str, sample_df: pd.DataFrame):
    """
    将 sample_df 的结构作为 background，缓存一个全局解释器。
    由于 DataFrame 不能直接作为 cache key，这里用 hash 字符串。
    """
    explainer = _make_explainer(model, sample_df)
    return explainer

# --- 2. 侧边栏导航 ---
st.sidebar.title("导航")
app_mode = st.sidebar.radio(
    "请选择功能模式",
    ["实时单次预测", "批量文件预测"]
)

# =====================================================================================
# --- 模式一：实时单次预测 ---
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
        try:
            prediction = float(model.predict(features_df)[0])
        except Exception as e:
            st.error(f"预测失败：{e}")
            st.stop()

        st.success(f"**预测的10年净现值 (NPV):  ¥ {prediction:,.2f}**")
        st.write("---")

        st.header("本次预测解释")
        st.write("下图展示了各个输入特征如何影响本次的预测结果。红色特征推动预测值升高，蓝色特征推动预测值降低。")

        try:
            explainer = _make_explainer(model, features_df)
            sv = explainer(features_df)  # 统一新接口：得到 Explanation
            # 使用 legacy force_plot 生成 HTML（兼容 streamlit 展示）
            fp = shap.force_plot(
                base_value=sv.base_values[0],
                shap_values=sv.values[0],
                features=features_df.iloc[0],
                matplotlib=False
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
            components.html(shap_html, height=220)
        except Exception as e:
            st.warning(f"生成解释图时出现问题（已使用回退接口或当前组合不支持 force_plot）：{e}")

# =====================================================================================
# --- 模式二：批量文件预测 ---
# =====================================================================================
elif app_mode == "批量文件预测":
    st.title("📂 批量文件预测")
    st.write("请上传 CSV 文件进行批量预测。文件需要包含与单次预测相同的 9 个特征列，**顺序也需一致**。")

    uploaded_file = st.file_uploader("选择一个 CSV 文件", type="csv")

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            # 验证上传文件的列名是否正确且顺序一致
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
                st.write("当前文件列顺序：")
                st.json(list(input_df.columns))

        except Exception as e:
            st.error(f"处理文件时出错：{e}")

# =====================================================================================
# --- 模型总体解读（全局重要性） ---
# =====================================================================================
st.write("---")
with st.expander("🔬 查看模型总体特征重要性"):
    st.write("""
    下图（SHAP 摘要图）展示了模型在全局范围内的行为。它告诉我们哪些特征对所有预测的平均影响最大。
    - **特征重要性**：Y 轴上的特征按重要性从上到下排序。
    - **影响方向**：点的颜色表示特征值的大小（红色为高，蓝色为低）。
    - **分布**：X 轴表示该特征对预测结果的影响（SHAP 值）。
    """)

    # 作为背景数据的样本（如无训练集，可先用演示样本）
    sample_data = {
        'solar_irradiance_kWh_m2_day': [5.048, 4.378, 5.217, 4.895, 3.688, 5.451, 5.022, 5.072, 3.756, 4.401],
        'system_size_kW':               [5.473, 5.773, 5.842, 5.007, 5.041, 5.600, 4.629, 5.675, 4.988, 4.232],
        'panel_efficiency_pct':         [16.50, 19.67, 17.03, 17.22, 21.06, 19.14, 23.37, 18.55, 21.81, 19.69],
        'inverter_efficiency_pct':      [97.36, 98.80, 95.85, 96.13, 98.06, 96.53, 97.68, 96.30, 95.08, 97.73],
        'install_cost_CNY_per_kW':      [8292, 6509, 8788, 7429, 5022, 8483, 6702, 5137, 5187, 7594],
        'o&m_CNY_per_kW_year':          [  55,  152,  193,  174,   92,   58,  191,  191,  130,  146],
        'electricity_price_CNY_per_kWh':[0.823, 0.717, 0.560, 0.786, 0.488, 0.798, 0.715, 0.500, 0.486, 0.788],
        'subsidy_CNY_per_kWh':          [0.083, 0.114, 0.000, 0.129, 0.144, 0.113, 0.099, 0.162, 0.136, 0.056],
        'discount_rate_pct':            [8.01,  5.50,  8.96,  4.69,  6.77,  4.29,  6.47,  4.07,  4.13,  7.29]
    }
    sample_df = pd.DataFrame(sample_data)

    try:
        # 解释器（带缓存）。用列名+行数做一个简易 hash key
        background_key = f"{','.join(sample_df.columns)}|{len(sample_df)}"
        explainer_global = get_explainer_for_background(background_key, sample_df)

        # 统一新接口
        sv_global = explainer_global(sample_df)

        # --- 修改部分 开始 ---
    
        # 1. 直接调用 SHAP 绘图，它会在 matplotlib 的“当前”图形上绘制
        #    我们不需要手动创建 fig, ax
        shap.summary_plot(sv_global.values, sample_df, show=False)
    
        # 2. 使用 plt.gcf() (Get Current Figure) 来获取 SHAP 刚刚绘制好的图形
        fig = plt.gcf()
    
        # 3. 将这个捕获到的图形传递给 streamlit
        st.pyplot(fig)

        # --- 修改部分 结束 ---

    except Exception as e:
        st.error(f"生成全局 SHAP 摘要图失败：{e}")
        st.info("提示：请检查 shap/xgboost/numpy 的版本是否相互兼容。建议固定依赖版本以避免云端环境变化导致的报错。")
