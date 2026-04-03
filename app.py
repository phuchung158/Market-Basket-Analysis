import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
    st.title("🛒 Phân tích hành vi mua sắm khách hàng")
    
    # 1. Thông tin bắt buộc
    with st.container(border=True):
        st.subheader("📝 Thông tin thực hiện")
        st.write("**Tên đề tài:** Phân tích hành vi mua sắm khách hàng và gợi ý sản phẩm dựa trên thuật toán Apriori & FP-Growth")
        st.write("**Họ tên SV:** Hà Thúc Phúc Hưng")
        st.write("**MSV:** 22T1020618")
        
    # 2. Giá trị thực tiễn
    st.subheader("💡 Giá trị thực tiễn")
    st.info("""Giải pháp này giúp các nhà quản lý bán lẻ hiểu rõ mối liên kết giữa các sản phẩm. 
    Từ đó tối ưu hóa việc sắp xếp kệ hàng (đặt các món hay mua cùng nhau gần nhau) và thiết kế 
    các chương trình khuyến mãi Combo hiệu quả để tăng doanh thu.""")

    # --- PHẦN BỔ SUNG: CÁC CHỈ SỐ TỔNG QUAN ---
    st.subheader("📊 Tổng quan bộ dữ liệu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số dòng dữ liệu", f"{len(df):,}")
    m2.metric("Số lượng sản phẩm", f"{df['itemDescription'].nunique()}")
    m3.metric("Số lượng giao dịch", f"{len(transactions):,}")

    # 3. Hiển thị dữ liệu thô
    st.subheader("🔍 1. Khám phá dữ liệu thô")
    st.write("Dữ liệu giao dịch mẫu (Lướt để xem chi tiết):")
    st.dataframe(df.head(100), use_container_width=True, height=300)

    # --- PHẦN BỔ SUNG: BẢNG LƯỚT LÊN XUỐNG THỐNG KÊ SẢN PHẨM ---
    st.subheader("📦 2. Danh mục sản phẩm & Số lượng bán")
    st.write("Bảng thống kê chi tiết tần suất xuất hiện của từng sản phẩm (Lướt để xem toàn bộ):")
    
    # Tính toán bảng thống kê
    product_stats = df['itemDescription'].value_counts().reset_index()
    product_stats.columns = ['Tên sản phẩm', 'Số lượng đã bán']
    
    # Hiển thị bảng có thanh cuộn (height=400 tạo ra thanh cuộn dọc)
    st.dataframe(
        product_stats, 
        use_container_width=True, 
        height=400, 
        hide_index=True
    )
    st.caption("💡 Bạn có thể nhấn vào tên cột để sắp xếp sản phẩm theo số lượng.")

    # 4. Biểu đồ phân tích
    st.subheader("📈 3. Biểu đồ phân tích trực quan")
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
    - **Quy mô:** Dữ liệu có hơn **38,000 dòng** ghi nhận giao dịch của khách hàng.
    - **Đặc trưng:** Có 3 cột chính (`Member_number`, `Date`, `itemDescription`). Trong đó `itemDescription` là đặc trưng quan trọng nhất để khai phá quy luật.
    - **Độ lệch:** Dữ liệu bị lệch về phía các mặt hàng thiết yếu như **Sữa (Whole milk)** và **Rau củ (Other vegetables)**. 
    - **Hành vi:** Đa số các giỏ hàng chỉ có từ **2-5 sản phẩm**, điều này cho thấy khách hàng thường mua lẻ tẻ, đòi hỏi thuật toán phải có ngưỡng hỗ trợ thấp để tìm ra sự liên kết.
    """)

# --- TRANG 2: TRIỂN KHAI MÔ HÌNH ---
elif page == "Trang 2: Triển khai dự báo":
    st.title("🎯 Triển khai Mô hình Dự báo Mua sắm")
    st.markdown("---")

    # 1. XỬ LÝ LOGIC: LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN
    # Đảm bảo bạn đã có thư mục models/ và file trained_model.pkl trong đó
    try:
        with open('models/file.pkl', 'rb') as f:
            model_rules = pickle.load(f)
        st.sidebar.success("✅ Đã kết nối với mô hình thành công!")
    except FileNotFoundError:
        st.sidebar.error("❌ Lỗi: Không tìm thấy file 'models/trained_model.pkl'. Hãy chạy file huấn luyện trước!")
        st.stop() # Dừng ứng dụng nếu không có mô hình

    # 2. THIẾT KẾ GIAO DIỆN NHẬP LIỆU (Sử dụng đa dạng Widget)
    st.subheader("📝 Nhập thông tin giao dịch")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Widget selectbox: Chọn sản phẩm (Danh mục)
            all_products = sorted(df['itemDescription'].unique())
            selected_item = st.selectbox("🛍️ Chọn sản phẩm khách đang cầm trên tay:", all_products)
            
            # Widget number_input: Số lượng gợi ý (Số)
            num_rec = st.number_input("🔢 Số lượng sản phẩm muốn gợi ý thêm:", min_value=1, max_value=5, value=2)

        with col2:
            # Widget slider: Ngưỡng tin cậy (Xác suất)
            conf_threshold = st.slider("🎯 Ngưỡng tin cậy tối thiểu (Confidence):", 0.01, 0.50, 0.10)
            
            # Widget text_input: Thông tin khách hàng (Văn bản)
            customer_name = st.text_input("👤 Tên khách hàng hoặc Mã thẻ:", "Khách hàng thân thiết")

    # 3. XỬ LÝ LOGIC DỰ BÁO & TIỀN XỬ LÝ
    st.markdown("### 🚀 Kết quả phân tích hành vi")
    
    if st.button("Thực hiện dự báo ngay"):
        with st.spinner('Mô hình đang tính toán...'):
            # 1. Lọc luật dựa trên sản phẩm đã chọn
            res = model_rules[model_rules['antecedents'].apply(lambda x: selected_item in x)]
            res = res[res['confidence'] >= conf_threshold]

            # 2. TIỀN XỬ LÝ: Loại bỏ các gợi ý trùng tên sản phẩm (Chỉ giữ lại cái tốt nhất)
            # Chúng ta sẽ biến đổi cột consequents từ frozenset sang string để dễ lọc
            res['suggested_name'] = res['consequents'].apply(lambda x: list(x)[0])
            res = res.sort_values(by='lift', ascending=False).drop_duplicates(subset=['suggested_name'])
            
            # Lấy số lượng theo yêu cầu của người dùng
            results = res.head(num_rec)

            if not results.empty:
                st.write(f"Hệ thống đề xuất cho **{customer_name}**:")
                
                # 3. HIỂN THỊ (Sử dụng đúng biến index i)
                for i in range(len(results)):
                    # Lấy dữ liệu của hàng hiện tại trong vòng lặp
                    current_row = results.iloc[i]
                    suggested_item = current_row['suggested_name']
                    confidence_val = current_row['confidence']
                    lift_val = current_row['lift']
                    
                    with st.expander(f"🌟 Gợi ý {i+1}: {suggested_item}", expanded=True):
                        col_result_1, col_result_2 = st.columns(2)
                        with col_result_1:
                            st.write(f"📦 Sản phẩm: **{suggested_item}**")
                            st.write(f"🔗 Độ liên quan (Lift): {lift_val:.2f}")
                        with col_result_2:
                            st.metric("Độ tin cậy", f"{confidence_val:.2%}")
                            st.progress(confidence_val)
                st.balloons()
            else:
                st.warning("Không tìm thấy gợi ý phù hợp.")
# --- TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG ---
# --- TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG ---
elif page == "Trang 3: Đánh giá & Hiệu năng":
    st.title("📊 Đánh giá & So sánh mô hình")

    try:
        # 1. Đo lường hiệu năng thời gian
        # Dùng một mẫu nhỏ hoặc cache để tránh treo máy
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        with st.spinner('Đang tính toán so sánh hiệu năng...'):
            t1 = time.time()
            apriori(df_encoded, min_support=0.002, use_colnames=True) # Tăng support lên cho nhanh
            time_apriori = time.time() - t1

            t2 = time.time()
            fpgrowth(df_encoded, min_support=0.002, use_colnames=True)
            time_fpgrowth = time.time() - t2

        # Hiển thị chỉ số
        st.subheader("1. Tốc độ xử lý (Vấn đề tối ưu)")
        col1, col2 = st.columns(2)
        col1.metric("Thời gian chạy Apriori", f"{time_apriori:.4f}s")
        col2.metric("Thời gian chạy FP-Growth", f"{time_fpgrowth:.4f}s")
        st.info("💡 Nhận xét: FP-Growth thường nhanh hơn khi dữ liệu lớn nhờ cấu trúc cây nén dữ liệu.")

        # 2. Biểu đồ kỹ thuật
        st.subheader("2. Phân tích các chỉ số đo lường")
        # Lấy lại rules để vẽ biểu đồ
        freq = fpgrowth(df_encoded, min_support=0.001, use_colnames=True)
        rules_eval = association_rules(freq, metric="lift", min_threshold=1)
        
        if not rules_eval.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            scatter = ax3.scatter(rules_eval['support'], rules_eval['confidence'], c=rules_eval['lift'], cmap='YlOrRd')
            plt.colorbar(scatter, label='Chỉ số Lift')
            ax3.set_xlabel('Độ hỗ trợ (Support)')
            ax3.set_ylabel('Độ tin cậy (Confidence)')
            st.pyplot(fig3)
        else:
            st.warning("Không đủ dữ liệu luật để vẽ biểu đồ.")

        # 3. Phân tích sai số
        st.subheader("3. Phân tích & Hướng cải thiện")
        st.write("- **Hạn chế:** Thuật toán nhạy cảm với các mặt hàng phổ biến, dễ tạo ra các luật 'hiển nhiên'.")
        st.write("- **Cải thiện:** Cần áp dụng thêm lọc luật theo giá trị kinh doanh (Profit) thay vì chỉ dựa vào tần suất.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải Trang 3: {e}")
