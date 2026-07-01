import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 設定網頁標題與排版
st.set_page_config(page_title="Bingham 泥流與明渠沖刷即時推估工具", layout="wide")

# 支持中文顯示
# 修正後的寫法：優先使用 Linux 伺服器常見的開源中文自行，再相容 Windows/Mac
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',      # Streamlit Cloud (Linux) 最常內建的中文自行
    'Noto Sans CJK JP',         # 另一種常見的 Linux 中文字形
    'Microsoft JhengHei',       # Windows 本地測試用
    'Arial'
]
plt.rcParams['axes.unicode_minus'] = False # 解決負號變亂碼/方塊的問題

st.title("🌊 Bingham 泥流與明渠沖刷即時互動推估工具")
st.markdown("本工具結合**降雨降水入滲流出模型**與 **Bingham 流體力學簡化公式**，即時估算特定地形下的沖刷流速、體積與極限流動距離。")

# ==========================================
# SIDEBAR: 輸入參數控制區
# ==========================================
st.sidebar.header("📐 1. 地形與坡度設置")
H = st.sidebar.slider("起點高程 H (m)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
L = st.sidebar.slider("水平距離 L (m)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)

# 計算坡度
slope_rad = np.arctan(H / L)
slope_deg = np.degrees(slope_rad)
slope_percent = (H / L) * 100

st.sidebar.markdown(f"**當前主坡度:** {slope_percent:.1f}% ({slope_deg:.1f}°)")

st.sidebar.header("型 2. 明渠斷面型態")
channel_type = st.sidebar.selectbox("斷面幾何類型", ["V 型河谷斷面", "矩形人工明渠"])
channel_w = st.sidebar.slider("渠道頂寬/底寬 W (m)", min_value=5.0, max_value=50.0, value=20.0)
if channel_type == "V 型河谷斷面":
    valley_m = st.sidebar.slider("V型邊坡係數 (1:m)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

 
st.sidebar.header("🌧️ 3. 降雨與集水區資料")
rain_mm = st.sidebar.slider("延時降雨量 (mm)", min_value=20, max_value=500, value=150, step=10)

# 這裡調整為公頃 (ha)，1 公頃 = 10,000 m²
catchment_ha = st.sidebar.slider("集水區面積 (公頃 ha)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
catchment_area = catchment_ha * 10000  # 自動換算為 m² 帶入後續力學計算
runoff_coef = st.sidebar.slider("地表逕流/流體啟動係數", min_value=0.2, max_value=0.9, value=0.6, step=0.05)

st.sidebar.header("🧪 4. Bingham 流體性質")
rho = st.sidebar.number_input("流體密度 rho (kg/m³)", value=1500.0, step=100.0)
tau_0 = st.sidebar.number_input("屈服應力 tau_0 (Pa)", value=35.0, step=5.0)
mu_p = st.sidebar.number_input("塑性粘度 mu_p (Pa·s)", value=0.8, step=0.1)

# ==========================================
# CORE CORE CORE: 理論核心計算
# ==========================================
g = 9.81

# 1. 體積推估 (降雨產生的總流體體積，考慮泥沙增量比)
sediment_bulking = 1.3  # 考慮沖刷帶來的泥沙體積放大效應
total_volume = (rain_mm / 1000.0) * catchment_area * runoff_coef * sediment_bulking

# 2. 水力幾何與流深估算 (簡化均勻流法)
# 假設洪峰持續時間內，斷面充填的代表性平均流深 h_e
if channel_type == "矩形人工明渠":
    h_e = np.cbrt(total_volume / (channel_w * L * 2)) # 估算合理水深範圍
    h_e = np.clip(h_e, 0.1, 5.0)
    A_flow = channel_w * h_e
    P_flow = channel_w + 2 * h_e
else: # V 型
    h_e = np.sqrt(total_volume / (valley_m * L * 2))
    h_e = np.clip(h_e, 0.1, 5.0)
    A_flow = valley_m * (h_e ** 2)
    P_flow = 2 * h_e * np.sqrt(1 + valley_m**2)

R_h = A_flow / P_flow # 水力半徑

# 3. Bingham 流速估算 (基於修正的 Buckingham 方程式或明渠 Bingham 簡化式)
# 重力沿坡向分力驅動剪切應力 tau_b = rho * g * R_h * sin(theta)
sin_theta = np.sin(slope_rad)
tau_b = rho * g * R_h * sin_theta

if tau_b <= tau_0:
    v_velocity = 0.0
    status_msg = "⚠️ 重力驅動應力小於屈服應力，流體無法啟動（發生停滯、堆積現象）。"
else:
    # Bingham 明渠流速簡化近似 (極限平衡與剪切速率積分)
    # V = (tau_b / mu_p) * (R_h / 3) * (1 - (1.5*(tau_0/tau_b)) + 0.5*(tau_0/tau_b)**3)
    v_velocity = (tau_b / mu_p) * (R_h / 3) * (1 - 1.5 * (tau_0 / tau_b) + 0.5 * (tau_0 / tau_b)**3)
    v_velocity = max(0.0, float(v_velocity))
    status_msg = "✅ 流體已克服屈服應力，正處於高速沖刷演進狀態。"

# 4. 沖刷流動距離推估 (Runout Distance)
# 當流體到達平緩地面（假設下遊平原坡度降為 0.5%）時的制動距離
slope_flat = 0.005
tau_b_flat = rho * g * h_e * slope_flat
# 動能轉化為克服屈服應力做功之簡化能量平衡
if v_velocity > 0:
    kinetic_energy = 0.5 * rho * v_velocity**2
    # 每單位長度抵抗力差值
    net_resist = max(1.0, tau_0 - tau_b_flat)
    runout_distance = L + (kinetic_energy * h_e) / (net_resist + 1e-3)
else:
    runout_distance = L

# ==========================================
# VISUALIZATION: 畫面上半部指標展現
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="📊 推估總流體體積", value=f"{total_volume:,.1f} m³")
col2.metric(label="⚡ 預期最大流速", value=f"{v_velocity:.2f} m/s")
col3.metric(label="🏁 總流動/運移距離", value=f"{runout_distance:.1f} m")
col4.metric(label="📐 代表性平均流深", value=f"{h_e:.2f} m")

st.info(status_msg)

# ==========================================
# PLOTS: 繪製地形剖面與渠道斷面
# ==========================================
st.subheader("📈 空間幾何與動態剖面可視化")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    # 繪製縱向地形坡度與流體剖面
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    x_profile = np.linspace(0, L, 100)
    z_profile = H - (H / L) * x_profile
    ax1.plot(x_profile, z_profile, color='gray', lw=2, label='原始河床地形')
    
    # 模擬流體覆蓋
    if v_velocity > 0:
        ax1.fill_between(x_profile, z_profile, z_profile + h_e, color='saddlebrown', alpha=0.7, label='Bingham 泥流')
    ax1.set_xlabel("水平距離 (m)")
    ax1.set_ylabel("高程 (m)")
    ax1.set_title("渠道縱向剖面示意圖")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

with col_plot2:
    # 繪製橫斷面型態
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if channel_type == "矩形人工明渠":
        x_c = [0, 0, channel_w, channel_w]
        y_c = [h_e * 1.5, 0, 0, h_e * 1.5]
        ax2.plot(x_c, y_c, color='black', lw=2)
        ax2.fill_between([0, channel_w], [0, 0], [h_e, h_e], color='saddlebrown', alpha=0.7, label='流體區域')
    else:
        # V型
        half_w = (h_e * 1.5) * valley_m
        x_c = [-half_w, 0, half_w]
        y_c = [h_e * 1.5, 0, h_e * 1.5]
        ax2.plot(x_c, y_c, color='black', lw=2)
        
        x_fluid = np.linspace(-h_e * valley_m, h_e * valley_m, 50)
        y_fluid = np.abs(x_fluid) / valley_m
        ax2.fill_between(x_fluid, y_fluid, h_e, color='saddlebrown', alpha=0.7, label='流體區域')
        
    ax2.set_xlabel("斷面寬度 (m)")
    ax2.set_ylabel("渠道深度 (m)")
    ax2.set_title(f"橫斷面幾何 ({channel_type})")
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

# ==========================================
# FOOTER: 理論說明與連動提示
# ==========================================
st.markdown("---")
st.markdown("""
### 💡 互動工具與 2D 數值模擬程式的連動說明
1. **本工具定位：** 採用一維（1D）簡化力學公式與集水區經驗公式，適合在**秒級**時間內進行大量「What-if」情境參數篩選。
2. **與 2D 程式接軌：** 當你在此工具中調整出危險的組合（例如：流速 > 5 m/s 且流動距離極遠），可將本工具右上側邊欄顯示的 `tau_0`, `mu_p`, `rho` 與推估出的**總流量/體積**，作為你 2D 偏微分方程模型的 `get_upstream_inflow(t)` 輸入邊界條件，進行更精確的河道氾濫與堆積範圍模擬。
""")
