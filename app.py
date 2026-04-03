import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Đồ án Market Basket Analysis", layout="wide")

# --- HÀM LOAD DỮ LIỆU (Dùng Cache để tăng tốc) ---
@st.cache_data
def get_data():
    df = pd.read_csv('data/Groceries_dataset.csv')
    # Tiền xử lý để tạo giỏ hàng
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()
    return df, transactions

df, transactions = get_data()

# --- THANH ĐIỀU HƯỚNG (SIDEBAR) ---
st.sidebar.title("📌 Danh mục dự án")
page = st.sidebar.radio("Chọn trang hiển thị:", 
    ["Trang 1: Giới thiệu & EDA", 
     "Trang 2: Triển khai mô hình", 
     "Trang 3: Đánh giá hiệu năng"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU (EDA)
# ---------------------------------------------------------
if page == "Trang 1: Giới thiệu & EDA":
    st.title("🚀 Phân tích quy luật mua sắm (Market Basket Analysis)")
    
    # Thông tin SV
    with st.expander("ℹ️ Thông tin sinh viên thực hiện", expanded=True):
        st.write("**Họ tên SV:** [Tên của bạn]")
        st.write("**MSSV:** [Mã số sinh viên của bạn]")
        st.write("**Đề tài:** Ứng dụng thuật toán Apriori tối ưu hóa gợi ý sản phẩm siêu thị.")

    st.subheader("1. Giá trị thực tiễn")
    st.info("Bài toán giúp doanh nghiệp hiểu thói quen mua sắm 'kèm theo' của khách hàng. Từ đó tối ưu việc sắp xếp kệ hàng, thiết kế các gói Combo khuyến mãi và xây dựng hệ thống gợi ý tự động nhằm tăng doanh thu.")

    st.subheader("2. Khám phá dữ liệu thô")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("3. Biểu đồ phân tích (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top 10 sản phẩm bán chạy nhất**")
        top_items = df['itemDescription'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_items.values, y=top_items.index, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**Tần suất mua sắm theo tháng**")
        df['Date'] = pd.to_datetime(df['Date'])
        monthly_counts = df.resample('ME', on='Date').size()
        fig2, ax2 = plt.subplots()
        monthly_counts.plot(kind='line', marker='o', ax=ax2, color='orange')
        st.pyplot(fig2)

    st.markdown("""
    **Nhận xét về dữ liệu:**
    - Dữ liệu bao gồm hơn 38,000 giao dịch đơn lẻ.
    - **Đặc trưng:** Có 3 cột chính là `Member_number`, `Date` và `itemDescription`.
    - **Độ lệch:** Dữ liệu tập trung mạnh vào các mặt hàng thiết yếu như Sữa tươi (Whole milk) và Rau củ. Đây là đặc trưng quan trọng để thiết lập ngưỡng *Support* phù hợp.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "Trang 2: Triển khai mô hình":
    st.title("🎯 Triển khai hệ thống gợi ý")
    
    st.sidebar.header("Cấu hình thuật toán")
    min_supp = st.sidebar.slider("Ngưỡng hỗ trợ (Support)", 0.001, 0.01, 0.001, format="%.3f")
    min_conf = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.01, 0.5, 0.1)

    # Xử lý Logic (Tiền xử lý & Apriori)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Chạy mô hình
    frequent_itemsets = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[rules['confidence'] >= min_conf]

    st.subheader("🎮 Trải nghiệm gợi ý thực tế")
    all_items = sorted(df['itemDescription'].unique())
    user_input = st.selectbox("Chọn sản phẩm khách hàng vừa bỏ vào giỏ:", all_items)

    if st.button("Dự đoán sản phẩm mua kèm"):
        # Tìm luật liên quan
        res = rules[rules['antecedents'].apply(lambda x: user_input in x)].sort_values(by='lift', ascending=False)
        
        if not res.empty:
            suggested = list(res.iloc[0]['consequents'])[0]
            conf = res.iloc[0]['confidence']
            lift = res.iloc[0]['lift']
            
            st.success(f"✅ Kết quả dự báo: Khách hàng khả năng cao sẽ mua thêm **{suggested}**")
            st.metric(label="Độ tin cậy (Confidence)", value=f"{conf*100:.2f}%")
            st.metric(label="Chỉ số Lift (Độ cải thiện)", value=f"{lift:.2f}")
        else:
            st.warning("Chưa tìm thấy luật kết hợp phù hợp cho sản phẩm này với ngưỡng hiện tại.")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
elif page == "Trang 3: Đánh giá hiệu năng":
    st.title("📊 Đánh giá chất lượng mô hình")

    # Tính toán lại rules để lấy thông số hiển thị
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    st.subheader("1. Các chỉ số đo lường đặc thù")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số luật tìm được", len(rules))
    c2.metric("Support trung bình", f"{rules['support'].mean():.4f}")
    c3.metric("Lift cao nhất", f"{rules['lift'].max():.2f}")

    st.subheader("2. Biểu đồ phân bố các chỉ số")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=rules, x="support", y="confidence", hue="lift", palette="magma", ax=ax3)
    plt.title("Mối quan hệ giữa Support, Confidence và Lift")
    st.pyplot(fig3)

    st.subheader("3. Phân tích sai số & Hướng cải thiện")
    st.markdown("""
    - **Sai số/Hạn chế:** Thuật toán thường gợi ý các mặt hàng quá phổ biến (như sữa tươi) cho mọi sản phẩm đầu vào vì tần suất xuất hiện của chúng quá cao trong tập dữ liệu.
    - **Trường hợp dự đoán yếu:** Các mặt hàng xa xỉ hoặc ít người mua (như đồ gia dụng đắt tiền) thường không đủ ngưỡng *Support* để sinh ra luật.
    - **Hướng cải thiện:** 1. Sử dụng thuật toán FP-Growth để tăng tốc độ xử lý.
        2. Kết hợp thêm thông tin cá nhân hóa khách hàng (tuổi, giới tính) thay vì chỉ dựa vào giỏ hàng đơn thuần.
    """)
