import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- CẤU HÌNH GIAO DIỆN & CHỮ TO ---
st.set_page_config(page_title="Đồ án Phân tích Giỏ hàng", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="View"] { font-size: 20px; }
    .stDataFrame div { font-size: 18px !important; }
    .stAlert p { font-size: 22px !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/Groceries_dataset.csv')
    # Gom nhóm tạo danh sách giỏ hàng
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()
    return df, transactions

df, transactions = load_data()

# --- THANH ĐIỀU HƯỚNG SIDEBAR ---
st.sidebar.title("📌 Danh mục nội dung")
page = st.sidebar.radio("Chọn trang hiển thị:", 
    ["Trang 1: Giới thiệu & EDA", 
     "Trang 2: Triển khai dự báo", 
     "Trang 3: Đánh giá & Hiệu năng"])

# --- TRANG 1: GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU (EDA) ---
if page == "Trang 1: Giới thiệu & EDA":
    st.title("🛒 Phân tích hành vi mua sắm khách hàng và gợi ý sản phẩm")
    
    # 1. Thông tin bắt buộc
    with st.container(border=True):
        st.subheader("📝 Thông tin thực hiện")
        st.write("**Tên đề tài:** Phân tích hành vi mua sắm khách hàng và gợi ý sản phẩm dựa trên thuật toán Apriori & FP-Growth")
        st.write("**Họ tên SV:** [Tên của bạn]")
        st.write("**MSSV:** [MSSV của bạn]")
        
    # 2. Giá trị thực tiễn
    st.subheader("💡 Giá trị thực tiễn")
    st.info("""Giải pháp này giúp các nhà quản lý bán lẻ hiểu rõ mối liên kết giữa các sản phẩm. 
    Từ đó tối ưu hóa việc sắp xếp kệ hàng (đặt các món hay mua cùng nhau gần nhau) và thiết kế 
    các chương trình khuyến mãi Combo hiệu quả để tăng doanh thu.""")

    # 3. Hiển thị dữ liệu thô
    st.subheader("📊 Khám phá dữ liệu (EDA)")
    st.write("Một phần dữ liệu giao dịch mẫu:")
    st.dataframe(df.head(10), use_container_width=True)

    # 4. Biểu đồ phân tích
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 sản phẩm xuất hiện nhiều nhất**")
        top_items = df['itemDescription'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_items.values, y=top_items.index, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**Phân bố số lượng sản phẩm mỗi hóa đơn**")
        basket_sizes = [len(t) for t in transactions]
        fig2, ax2 = plt.subplots()
        sns.histplot(basket_sizes, bins=10, kde=True, color='orange', ax=ax2)
        ax2.set_xlabel("Số sản phẩm trong 1 giỏ hàng")
        st.pyplot(fig2)

    # 5. Giải thích dữ liệu
    st.markdown("""
    **Nhận xét về dữ liệu:**
    - **Quy mô:** Dữ liệu có hơn 38,000 dòng ghi nhận giao dịch của khách hàng.
    - **Đặc trưng:** Có 3 cột chính (`Member_number`, `Date`, `itemDescription`). Trong đó `itemDescription` là đặc trưng quan trọng nhất để khai phá quy luật.
    - **Độ lệch:** Dữ liệu bị lệch về phía các mặt hàng thiết yếu (Sữa, rau củ). Đa số các giỏ hàng chỉ có từ 2-5 sản phẩm, điều này đòi hỏi ngưỡng *Support* phải đặt thấp mới tìm được quy luật.
    """)

# --- TRANG 2: TRIỂN KHAI MÔ HÌNH ---
elif page == "Trang 2: Triển khai dự báo":
    st.title("🎯 Gợi ý sản phẩm thông minh")
    
    # Thiết kế giao diện nhập liệu
    st.sidebar.header("Cấu hình mô hình")
    algo_choice = st.sidebar.selectbox("Chọn thuật toán chạy:", ["Apriori", "FP-Growth"])
    min_supp = st.sidebar.number_input("Ngưỡng hỗ trợ (Support)", 0.001, 0.01, 0.001, format="%.3f")
    min_conf = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.05, 0.5, 0.1)

    # Xử lý logic tiền xử lý & chạy mô hình
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Chạy thuật toán được chọn
    if algo_choice == "Apriori":
        frequent_itemsets = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_supp, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[rules['confidence'] >= min_conf]

    # Giao diện tương tác
    st.subheader("🎮 Trải nghiệm dự đoán mua kèm")
    all_products = sorted(df['itemDescription'].unique())
    user_item = st.selectbox("Khách hàng đang mua sản phẩm:", all_products)

    if st.button("Dự đoán sản phẩm mua thêm"):
        if not rules.empty:
            res = rules[rules['antecedents'].apply(lambda x: user_item in x)].sort_values(by='lift', ascending=False)
            if not res.empty:
                best_match = list(res.iloc[0]['consequents'])[0]
                lift_score = res.iloc[0]['lift']
                conf_score = res.iloc[0]['confidence']
                
                st.success(f"✅ Dự báo: Khách hàng thường mua kèm **{best_match}**")
                st.metric("Độ tin cậy (Confidence)", f"{conf_score:.2%}")
                st.write(f"👉 Ý nghĩa: Khả năng mua {best_match} tăng gấp **{lift_score:.2f} lần** khi có {user_item} trong giỏ.")
            else:
                st.warning("Chưa có quy luật phù hợp cho sản phẩm này với cấu hình hiện tại.")
        else:
            st.error("Không tìm thấy quy luật nào. Hãy giảm ngưỡng Support hoặc Confidence!")

# --- TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG ---
elif page == "Trang 3: Đánh giá hiệu năng":
    st.title("📊 Đánh giá & So sánh mô hình")

    # 1. Đo lường hiệu năng thời gian
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    t1 = time.time()
    apriori(df_encoded, min_support=0.001, use_colnames=True)
    time_apriori = time.time() - t1

    t2 = time.time()
    fpgrowth(df_encoded, min_support=0.001, use_colnames=True)
    time_fpgrowth = time.time() - t2

    # Hiển thị chỉ số
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Thời gian chạy Apriori", f"{time_apriori:.4f}s")
    with col2:
        st.metric("Thời gian chạy FP-Growth", f"{time_fpgrowth:.4f}s")

    # 2. Biểu đồ kỹ thuật
    st.subheader("📈 Phân tích quy luật tìm được")
    # Lấy lại rules để vẽ biểu đồ
    freq = fpgrowth(df_encoded, min_support=0.001, use_colnames=True)
    rules_eval = association_rules(freq, metric="lift", min_threshold=1)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    plt.scatter(rules_eval['support'], rules_eval['confidence'], c=rules_eval['lift'], cmap='YlOrRd')
    plt.colorbar(label='Chỉ số Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Mối liên quan giữa các chỉ số đo lường')
    st.pyplot(fig3)

    # 3. Phân tích sai số
    st.subheader("⚠️ Phân tích sai số & Hướng cải thiện")
    st.markdown("""
    - **Điểm yếu:** Mô hình thường gợi ý các mặt hàng "đại trà" (sữa, rau củ) do tần suất xuất hiện quá cao trong dữ liệu, làm mờ đi các quy luật đặc thù.
    - **Sai số:** Một số sản phẩm hiếm (ngách) sẽ không bao giờ được gợi ý nếu đặt ngưỡng Support cứng nhắc.
    - **Hướng cải thiện:** 1. Áp dụng thuật toán **FP-Growth** thay vì Apriori để tăng tốc độ khi dữ liệu lớn.
        2. Sử dụng thêm các yếu tố cá nhân hóa (theo Member_number) thay vì chỉ dựa vào giỏ hàng chung.
    """)
