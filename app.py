import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Đồ án Phân tích Giỏ hàng", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="View"] { font-size: 20px; }
    .stDataFrame div { font-size: 18px !important; }
    .stAlert p { font-size: 22px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    # Đảm bảo file csv nằm trong thư mục data/
    df = pd.read_csv('data/Groceries_dataset.csv')
    # Gom nhóm tạo danh sách giỏ hàng
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()
    return df, transactions

try:
    df, transactions = load_data()
except Exception as e:
    st.error(f"Không tìm thấy dữ liệu: {e}")
    st.stop()

# --- 3. THANH ĐIỀU HƯỚNG SIDEBAR ---
st.sidebar.title("📌 Danh mục nội dung")
page = st.sidebar.radio("Chọn trang hiển thị:", 
    ["Giới thiệu & EDA", 
     "Triển khai dự báo", 
     "Đánh giá & Hiệu năng"])

# --- TRANG 1: GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU (EDA) ---
if page == "Giới thiệu & EDA":
    st.title("🛒 Phân tích hành vi mua sắm khách hàng")
    
    with st.container(border=True):
        st.subheader("📝 Thông tin thực hiện")
        st.write("**Tên đề tài:** Phân tích hành vi mua sắm và gợi ý sản phẩm dựa trên thuật toán FP-Growth nhằm nâng cao hiệu quả kinh doanh")
        st.write("**Họ tên SV:** Hà Thúc Phúc Hưng")
        st.write("**MSV:** 22T1020618")
        
    st.subheader("💡 Giá trị thực tiễn")
    st.info("""Giải pháp này giúp các nhà quản lý bán lẻ hiểu rõ mối liên kết giữa các sản phẩm. 
    Từ đó tối ưu hóa việc sắp xếp kệ hàng và thiết kế các chương trình khuyến mãi Combo hiệu quả.""")

    st.subheader("📊 Tổng quan bộ dữ liệu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số dòng dữ liệu", f"{len(df):,}")
    m2.metric("Số lượng sản phẩm", f"{df['itemDescription'].nunique()}")
    m3.metric("Số lượng giao dịch", f"{len(transactions):,}")

    st.subheader("🔍 1. Khám phá dữ liệu thô")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📦 2. Danh mục sản phẩm & Tra cứu số lượng")
    product_stats = df['itemDescription'].value_counts().reset_index()
    product_stats.columns = ['Tên sản phẩm', 'Số lượng đã bán']
    
    st.dataframe(
        product_stats, 
        use_container_width=True, 
        height=300, 
        hide_index=True,
        column_config={
            "Tên sản phẩm": st.column_config.TextColumn("Tên sản phẩm"),
            "Số lượng đã bán": st.column_config.NumberColumn("Số lượng đã bán", format="%d 🛒")
        }
    )

    st.subheader("📈 3. Biểu đồ phân tích trực quan")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 sản phẩm bán chạy nhất**")
        fig, ax = plt.subplots()
        sns.barplot(x='Số lượng đã bán', y='Tên sản phẩm', data=product_stats.head(10), palette='viridis', ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("**Phân bố số lượng sản phẩm mỗi hóa đơn**")
        basket_sizes = [len(t) for t in transactions]
        fig2, ax2 = plt.subplots()
        sns.histplot(basket_sizes, bins=10, kde=True, color='orange', ax=ax2)
        st.pyplot(fig2)

# --- TRANG 2: TRIỂN KHAI MÔ HÌNH ---
elif page == "Triển khai dự báo":
    st.title("🎯 Triển khai Mô hình Dự báo Mua sắm")
    
    try:
        with open('models/file.pkl', 'rb') as f:
            model_rules = pickle.load(f)
        st.sidebar.success("✅ Đã kết nối mô hình thành công!")
    except:
        st.error("❌ Không tìm thấy file 'models/file.pkl'.")
        st.stop()

    st.subheader("📝 Nhập thông tin giao dịch")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            all_products = sorted(df['itemDescription'].unique())
            selected_item = st.selectbox("🛍️ Chọn sản phẩm khách mua:", all_products)
            num_rec = st.number_input("🔢 Số lượng gợi ý:", 1, 5, 2)
        with col2:
            conf_threshold = st.slider("🎯 Ngưỡng tin cậy (Confidence):", 0.01, 0.50, 0.05)
            customer_name = st.text_input("👤 Tên khách hàng:", "Khách hàng thân thiết")

    if st.button("Thực hiện dự báo ngay"):
        input_set = {selected_item}
        res = model_rules[model_rules['antecedents'].apply(lambda x: input_set.issubset(x))]
        res = res[res['confidence'] >= conf_threshold]
        res['suggested_name'] = res['consequents'].apply(lambda x: list(x)[0])
        results = res.sort_values(by='lift', ascending=False).drop_duplicates(subset=['suggested_name']).head(num_rec)

        if not results.empty:
            for i in range(len(results)):
                row = results.iloc[i]
                with st.expander(f"🌟 Gợi ý {i+1}: {row['suggested_name']}", expanded=True):
                    c_res1, c_res2 = st.columns(2)
                    c_res1.write(f"🔗 Độ liên quan (Lift): {row['lift']:.2f}")
                    c_res2.metric("Độ tin cậy", f"{row['confidence']:.2%}")
                    st.progress(row['confidence'])
            st.balloons()
        else:
            st.warning("Không tìm thấy gợi ý phù hợp với ngưỡng này.")

# --- TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG ---
elif page == "Đánh giá & Hiệu năng":
    st.title("📊 Đánh giá & Hiệu năng Mô hình")
    
    try:
        with open('models/file.pkl', 'rb') as f:
            rules_eval = pickle.load(f)

        st.subheader("1. Các chỉ số đo lường chất lượng luật")
        with st.container(border=True):
            # Căn chỉnh 3 cột thẳng hàng cho cả 2 hàng
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số luật", len(rules_eval))
            c2.metric("Độ tin cậy TB", f"{rules_eval['confidence'].mean():.2%}")
            c3.metric("Độ tương quan TB (Lift)", f"{rules_eval['lift'].mean():.2f}")
            
            c1.metric("Độ tin cậy cao nhất", f"{rules_eval['confidence'].max():.2%}")
            c2.metric("Số luật Lift > 2.0", len(rules_eval[rules_eval['lift'] > 2]))

        st.subheader("2. Ma trận tương quan (Heatmap) - Top 20 luật tốt nhất")
        top_20 = rules_eval.sort_values(by='lift', ascending=False).head(20).copy()
        top_20['ant'] = top_20['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_20['con'] = top_20['consequents'].apply(lambda x: ', '.join(list(x)))
        pivot = top_20.pivot(index='ant', columns='con', values='lift')
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax3)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig3)

        st.subheader("🚀 Hướng cải thiện & Phát triển")
        with st.container(border=True):
            col_im1, col_im2 = st.columns(2)
            with col_im1:
                st.markdown("**1. Phân đoạn khách hàng:** Chia nhóm khách hàng để tìm luật đặc thù.")
                st.markdown("**2. Tối ưu thời gian:** Phân tích hành vi theo mùa/giờ.")
            with col_im2:
                st.markdown("**3. Hệ thống Hybrid:** Kết hợp với Collaborative Filtering.")
                st.markdown("**4. Gom nhóm sản phẩm:** Giảm độ thưa, tăng Confidence.")
        
        st.success("Mô hình đã hoàn thành mục tiêu khai phá quy luật cơ bản.")

    except Exception as e:
        st.error(f"Lỗi hiển thị đánh giá: {e}")
