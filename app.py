import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="So sánh Apriori & FP-Growth", layout="wide")

# --- CSS TÙY CHỈNH CHỮ TO ---
st.markdown("""
    <style>
    html, body, [class*="View"] { font-size: 18px; }
    .stDataFrame div { font-size: 16px !important; }
    .stAlert p { font-size: 20px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DỮ LIỆU ---
@st.cache_data
def get_data():
    df = pd.read_csv('data/Groceries_dataset.csv')
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()
    return df, transactions

df, transactions = get_data()

# --- SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("📌 Menu Dự Án")
page = st.sidebar.radio("Chọn trang:", ["EDA & Giới thiệu", "So sánh Thuật toán", "Gợi ý thông minh"])

# THÔNG SỐ CHUNG TRÊN SIDEBAR
st.sidebar.header("Cấu hình thuật toán")
min_supp = st.sidebar.slider("Ngưỡng hỗ trợ (Support)", 0.001, 0.01, 0.001, format="%.3f")
min_conf = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.01, 0.5, 0.1)

# --- TRANG 1: EDA ---
if page == "EDA & Giới thiệu":
    st.title("📊 Khám phá dữ liệu & Giới thiệu")
    st.info("**Đề tài:** So sánh hiệu năng Apriori và FP-Growth trong khai phá luật kết hợp.")
    st.subheader("Dữ liệu thô")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 sản phẩm phổ biến**")
        fig, ax = plt.subplots()
        df['itemDescription'].value_counts().head(10).plot(kind='bar', ax=ax, color='skyblue')
        st.pyplot(fig)
    with col2:
        st.write("**Số lượng hóa đơn theo thời gian**")
        df['Date'] = pd.to_datetime(df['Date'])
        fig2, ax2 = plt.subplots()
        df.resample('ME', on='Date').size().plot(ax=ax2, color='green')
        st.pyplot(fig2)

# --- TRANG 2: SO SÁNH THUẬT TOÁN ---
elif page == "So sánh Thuật toán":
    st.title("⚖️ So sánh Apriori vs FP-Growth")
    
    # Tiền xử lý ma trận
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 1. Chạy Apriori
    start_time = time.time()
    freq_apriori = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    apriori_time = time.time() - start_time

    # 2. Chạy FP-Growth
    start_time = time.time()
    freq_fpgrowth = fpgrowth(df_encoded, min_support=min_supp, use_colnames=True)
    fpgrowth_time = time.time() - start_time

    # Hiển thị kết quả so sánh
    col1, col2 = st.columns(2)
    col1.metric("Thời gian chạy Apriori", f"{apriori_time:.4f} giây")
    col2.metric("Thời gian chạy FP-Growth", f"{fpgrowth_time:.4f} giây")

    # Biểu đồ so sánh
    st.subheader("Biểu đồ so sánh tốc độ")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    algos = ['Apriori', 'FP-Growth']
    times = [apriori_time, fpgrowth_time]
    sns.barplot(x=times, y=algos, palette='coolwarm', ax=ax3)
    st.pyplot(fig3)

    st.markdown(f"""
    **Nhận xét:** - FP-Growth thường chạy **nhanh hơn** Apriori vì nó sử dụng cấu trúc cây (FP-Tree) thay vì phải quét dữ liệu nhiều lần và tạo ứng viên (candidates) như Apriori.
    - Cả hai thuật toán đều cho ra **{len(freq_apriori)} tập mục phổ biến** giống hệt nhau.
    """)

# --- TRANG 3: GỢI Ý ---
elif page == "Gợi ý thông minh":
    st.title("🔍 Demo Gợi ý thực tế")
    # Sử dụng kết quả từ FP-Growth cho nhanh
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    freq_itemsets = fpgrowth(df_encoded, min_support=min_supp, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
    rules = rules[rules['confidence'] >= min_conf]

    all_items = sorted(df['itemDescription'].unique())
    user_input = st.selectbox("Khách hàng đang chọn sản phẩm:", all_items)

    if st.button("Dự đoán"):
        res = rules[rules['antecedents'].apply(lambda x: user_input in x)].sort_values(by='lift', ascending=False)
        if not res.empty:
            suggested = list(res.iloc[0]['consequents'])[0]
            st.success(f"💡 Gợi ý mua kèm: **{suggested}**")
            st.write(f"Độ tin cậy: {res.iloc[0]['confidence']:.2%}")
        else:
            st.warning("Không tìm thấy luật phù hợp.")
