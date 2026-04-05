import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- 1. CẤU HÌNH GIAO DIỆN & CSS TỔNG HỢP ---
st.set_page_config(page_title="Đồ án Phân tích Giỏ hàng", layout="wide")

st.markdown("""
    <style>
    /* Chỉnh cỡ chữ và nền */
    html, body, [class*="View"] { font-size: 18px; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    /* Style cho Card kết quả ở trang Dự báo */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    .badge {
        background: linear-gradient(45deg, #4B0082, #8A2BE2);
        color: white; padding: 5px 12px; border-radius: 10px;
        font-size: 0.8rem; font-weight: bold;
    }
    .food-title { color: #1a1a1a; font-size: 1.1rem; font-weight: 700; margin-left: 10px; }
    .metric-container { display: flex; justify-content: space-between; margin-top: 15px; gap: 10px; }
    .metric-item {
        background: #f8f9fa; flex: 1; padding: 10px;
        border-radius: 12px; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/Groceries_dataset.csv')
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()
    return df, transactions

df, transactions = load_data()

# --- 3. THANH ĐIỀU HƯỚNG SIDEBAR ---
st.sidebar.title("📌 Danh mục nội dung")
page = st.sidebar.radio("Chọn trang hiển thị:", 
    [" Giới thiệu & EDA", 
     " Triển khai dự báo", 
     " Đánh giá & Hiệu năng"])

# ==========================================
# TRANG 1: GIỚI THIỆU & EDA
# ==========================================
if page == " Giới thiệu & EDA":
    st.title("🛒 Phân tích hành vi mua sắm khách hàng")
    
    with st.container(border=True):
        st.subheader("📝 Thông tin thực hiện")
        st.write("**Tên đề tài:** Phân tích hành vi mua sắm và gợi ý sản phẩm dựa trên thuật toán FP-Growth")
        st.write("**Sinh viên:** Hà Thúc Phúc Hưng | **MSV:** 22T1020618")

    st.subheader("📊 Tổng quan bộ dữ liệu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng dòng dữ liệu", f"{len(df):,}")
    m2.metric("Số lượng sản phẩm", f"{df['itemDescription'].nunique()}")
    m3.metric("Số lượng giao dịch", f"{len(transactions):,}")

    st.subheader("🔍 Khám phá dữ liệu")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    product_stats = df['itemDescription'].value_counts().reset_index()
    product_stats.columns = ['Tên sản phẩm', 'Số lượng']
    
    with col1:
        st.write("**Top 10 sản phẩm bán chạy**")
        fig, ax = plt.subplots()
        sns.barplot(x='Số lượng', y='Tên sản phẩm', data=product_stats.head(10), palette='viridis', ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("**Phân bố giỏ hàng**")
        basket_sizes = [len(t) for t in transactions]
        fig2, ax2 = plt.subplots()
        sns.histplot(basket_sizes, bins=10, kde=True, color='orange', ax=ax2)
        st.pyplot(fig2)

# ==========================================
# TRANG 2: TRIỂN KHAI DỰ BÁO
# ==========================================
elif page == " Triển khai dự báo":
    st.title("🎯 Dự báo & Gợi ý Sản phẩm")
    
    try:
        with open('models/file.pkl', 'rb') as f:
            model_rules = pickle.load(f)
        st.sidebar.success("✅ Đã kết nối mô hình")
    except FileNotFoundError:
        st.error("❌ Thiếu file models/file.pkl")
        st.stop()

    # Form nhập liệu ở Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Tìm kiếm & Lọc")
        all_products = sorted(df['itemDescription'].unique())
        selected_item = st.selectbox("🛍️ Chọn sản phẩm khách mua:", all_products)
        num_rec = st.number_input("🔢 Số lượng gợi ý:", 1, 10, 4)
        conf_threshold = st.slider("🎯 Ngưỡng tin cậy (Confidence):", 0.01, 0.50, 0.05)
        predict_btn = st.button("🚀 THỰC HIỆN DỰ BÁO", use_container_width=True)

    if predict_btn:
        st.subheader(f"✨ Gợi ý cho sản phẩm: {selected_item}")
        input_set = {selected_item}
        res = model_rules[model_rules['antecedents'].apply(lambda x: input_set.issubset(x))]
        res = res[res['confidence'] >= conf_threshold]
        res['suggested_name'] = res['consequents'].apply(lambda x: list(x)[0])
        results = res.sort_values(by='lift', ascending=False).drop_duplicates('suggested_name').head(num_rec)

        if not results.empty:
            for i in range(0, len(results), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(results):
                        row = results.iloc[i + j]
                        with cols[j]:
                            st.markdown(f"""
                            <div class="result-card">
                                <span class="badge">#{i + j + 1}</span>
                                <span class="food-title">{row['suggested_name'].upper()}</span>
                                <div class="metric-container">
                                    <div class="metric-item">
                                        <p style="margin:0; font-size:10px;">ĐỘ TIN CẬY</p>
                                        <p style="margin:0; font-size:16px; color:#2ecc71; font-weight:bold;">{row['confidence']:.1%}</p>
                                    </div>
                                    <div class="metric-item">
                                        <p style="margin:0; font-size:10px;">CHỈ SỐ LIFT</p>
                                        <p style="margin:0; font-size:16px; color:#3498db; font-weight:bold;">{row['lift']:.2f}</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.warning("Không tìm thấy gợi ý phù hợp.")

# ==========================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ==========================================
elif page == " Đánh giá & Hiệu năng":
    st.title("📊 Đánh giá Hiệu năng Mô hình")
    
    try:
        with open('models/file.pkl', 'rb') as f:
            rules_eval = pickle.load(f)
        
        # 1. HIỂN THỊ CÁC CHỈ SỐ DASHBOARD (Khớp với báo cáo của bạn)
        # Tạo 3 cột để các chỉ số thẳng hàng nhau
        col_m1, col_m2, col_m3 = st.columns(3)
    
        # Hàng 1
        col_m1.metric("Tổng số luật tìm được", f"{len(rules_eval)}")
        col_m2.metric("Độ tin cậy TB (Confidence)", f"{rules_eval['confidence'].mean():.2%}")
        col_m3.metric("Độ tương quan TB (Lift)", f"{rules_eval['lift'].mean():.2f}")

        # Hàng 2 (Dùng lại các cột cũ để thẳng hàng)
        col_m1.metric("Độ tin cậy cao nhất", f"{rules_eval['confidence'].max():.2%}")
        col_m2.metric("Số luật có Lift > 2.0", f"{len(rules_eval[rules_eval['lift'] > 2])}")
        # col_m3 có thể để trống hoặc thêm một chỉ số khác nếu muốn
        st.divider()

        # 2. BIỂU ĐỒ SCATTER PLOT
        st.subheader("📈 Phân tích tương quan giữa các chỉ số")
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(rules_eval['support'], rules_eval['confidence'], 
                           c=rules_eval['lift'], cmap='YlOrRd', alpha=0.6)
        plt.colorbar(scatter, label='Chỉ số Lift')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Tương quan giữa Support, Confidence và Lift')
        st.pyplot(fig) #

        # 3. BIỂU ĐỒ HEATMAP (Đã sửa lỗi duplicate index)
        st.subheader("🔥 Heatmap: Mối tương quan Top 20 luật")
        # Lấy top 20, chuyển sang string và xử lý trùng lặp trước khi pivot
        top_20 = rules_eval.sort_values(by='lift', ascending=False).head(20).copy()
        top_20['ant'] = top_20['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_20['con'] = top_20['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # KHẮC PHỤC LỖI: Groupby để loại bỏ các cặp trùng lặp trước khi pivot
        pivot_data = top_20.groupby(['ant', 'con'])['lift'].mean().unstack()

        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax3)
        ax3.set_title('Ma trận tương quan Lift giữa các cặp sản phẩm')
        st.pyplot(fig3)

        # 4. HƯỚNG CẢI THIỆN
        st.subheader("🚀 Hướng cải thiện")
        st.success("""
        1. **Phân đoạn khách hàng (Clustering):** Chia khách hàng thành các nhóm (Ví dụ: Nhóm thích đồ ngọt, nhóm nội trợ) trước khi chạy FP-Growth để có luật chính xác hơn cho từng nhóm.
        2. **Bổ sung biến thời gian:** Phân tích theo mùa (Tết mua gì, Hè mua gì) để gợi ý mang tính thời điểm cao hơn.
        3. **Dữ liệu lớn hơn:** Thu thập thêm lịch sử giao dịch để tăng độ tin cậy cho các mặt hàng ít phổ biến.
        """)
    except FileNotFoundError:
        st.error("⚠️ Vui lòng chạy huấn luyện mô hình để có dữ liệu đánh giá!")
