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
    [" Giới thiệu & EDA", 
     " Triển khai dự báo", 
     " Đánh giá & Hiệu năng"])

# --- GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU (EDA) ---
if page == " Giới thiệu & EDA":
    st.title("🛒 Phân tích hành vi mua sắm khách hàng")
    
    # 1. Thông tin bắt buộc
    with st.container(border=True):
        st.subheader("📝 Thông tin thực hiện")
        st.write("**Tên đề tài:** Phân tích hành vi mua sắm và gợi ý sản phẩm dựa trên thuật toán FP-Growth nhằm nâng cao hiệu quả kinh doanh")
        st.write("**Họ tên SV:** Hà Thúc Phúc Hưng")
        st.write("**MSV:** 22T1020618")
        
    # 2. Giá trị thực tiễn
    st.subheader("💡 Giá trị thực tiễn")
    st.info("""Giải pháp này giúp các nhà quản lý bán lẻ hiểu rõ mối liên kết giữa các sản phẩm. 
    Từ đó có thể chủ động gợi ý sản phẩm đi kèm. Khi khách hàng mua món A, hệ thống sẽ tự động "nhắc" món B, từ đó tăng kích thước giỏ hàng và doanh thu trên mỗi hóa đơn..""")

    # --- CÁC CHỈ SỐ TỔNG QUAN ---
    st.subheader("📊 Tổng quan bộ dữ liệu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số dòng dữ liệu", f"{len(df):,}")
    m2.metric("Số lượng sản phẩm", f"{df['itemDescription'].nunique()}")
    m3.metric("Số lượng giao dịch", f"{len(transactions):,}")

    # 3. Hiển thị dữ liệu thô 
    st.subheader("🔍 1. Khám phá dữ liệu thô")
    st.write("Mẫu 10 dòng giao dịch đầu tiên trong tập dữ liệu:")
    st.dataframe(df.head(10), use_container_width=True) # Đã chỉnh về 10 dòng

    # 4. BẢNG THỐNG KÊ SẢN PHẨM 
    st.subheader("📦 2. Danh mục sản phẩm & Tra cứu số lượng")
    st.write("Sử dụng biểu tượng 🔍 ở góc trên bên phải bảng để tìm kiếm nhanh tên sản phẩm:")
    
    # Tính toán bảng thống kê
    product_stats = df['itemDescription'].value_counts().reset_index()
    product_stats.columns = ['Tên sản phẩm', 'Số lượng đã bán']
    
    # Hiển thị bảng có thanh cuộn và tích hợp tìm kiếm mặc định của Streamlit
    st.dataframe(
        product_stats, 
        use_container_width=True, 
        height=400, 
        hide_index=True,
        # Kích hoạt tính năng tìm kiếm và lọc trên các cột
        column_config={
            "Tên sản phẩm": st.column_config.TextColumn("Tên sản phẩm", help="Gõ để tìm kiếm..."),
            "Số lượng đã bán": st.column_config.NumberColumn("Số lượng đã bán", format="%d 🛒")
        }
    )

    # 5. Biểu đồ phân tích trực quan
    st.subheader("📈 3. Biểu đồ phân tích")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 sản phẩm bán chạy nhất**")
        top_items = product_stats.head(10)
        fig, ax = plt.subplots()
        sns.barplot(x='Số lượng đã bán', y='Tên sản phẩm', data=top_items, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**Phân bố số lượng sản phẩm mỗi hóa đơn**")
        basket_sizes = [len(t) for t in transactions]
        fig2, ax2 = plt.subplots()
        sns.histplot(basket_sizes, bins=10, kde=True, color='orange', ax=ax2)
        ax2.set_xlabel("Số sản phẩm / Giỏ hàng")
        st.pyplot(fig2)

  # 6. Giải thích & Nhận xét dữ liệu (Phần bắt buộc)
    st.subheader("📝 Nhận xét về đặc trưng dữ liệu")
    
    with st.container(border=True):
        st.markdown(f"""
        **1. Về các đặc trưng quan trọng:**
        - Bộ dữ liệu này có **3 đặc trưng chính**: `Member_number` (Mã khách hàng), `Date` (Ngày giao dịch) và `itemDescription` (Tên sản phẩm).
        - Trong bài toán khai phá luật kết hợp, **`itemDescription`** là đặc trưng quan trọng nhất vì nó chứa thông tin về hành vi mua sắm. 
        - Hai đặc trưng còn lại đóng vai trò là "khóa" để gom nhóm các sản phẩm vào cùng một giỏ hàng (Transaction).

        **2. Về độ lệch của dữ liệu (Data Skewness):**
        - Dữ liệu có hiện tượng **lệch rất mạnh** về phía các mặt hàng nhu yếu phẩm. Ví dụ: *Whole milk* xuất hiện nhiều gấp 10-20 lần các mặt hàng ngách như *Specialty chocolate* hay *Frozen chicken*.
        - **Ảnh hưởng:** Độ lệch này khiến các quy luật liên quan đến Sữa hoặc Rau củ sẽ có độ hỗ trợ (Support) rất cao, trong khi các quy luật thú vị ở các mặt hàng khác dễ bị bỏ qua nếu ta đặt ngưỡng Support quá lớn.

        **3. Cấu trúc giỏ hàng:**
        - Với tổng cộng **{df['itemDescription'].nunique()}** loại sản phẩm nhưng trung bình mỗi hóa đơn chỉ có **2-5 món**, tập dữ liệu này được đánh giá là "Dữ liệu thưa" (Sparse Data). 
        - Điều này giải thích tại sao chúng ta cần sử dụng thuật toán mạnh như **FP-Growth** để xử lý hiệu quả hơn so với Apriori truyền thống.
        """)

# --- TRIỂN KHAI MÔ HÌNH ---
elif page == " Triển khai dự báo":
    st.title("🎯 Triển khai Mô hình Dự báo Mua sắm")
    st.markdown("---")

    # 1. XỬ LÝ LOGIC: LOAD MÔ HÌNH ĐÃ HUẤN LUYỆN
    # Đảm bảo bạn đã có thư mục models/ và file trained_model.pkl trong đó
    try:
        with open('models/file.pkl', 'rb') as f:
            model_rules = pickle.load(f)
        st.sidebar.success("✅ Đã kết nối với mô hình thành công!")
    except FileNotFoundError:
        st.sidebar.error("❌ Lỗi: Không tìm thấy file 'models/file.pkl'. Hãy chạy file huấn luyện trước!")
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
            conf_threshold = st.slider("🎯 Ngưỡng tin cậy tối thiểu (Confidence):", 0.01, 0.50, 0.05)
            
            # Widget text_input: Thông tin khách hàng (Văn bản)
            customer_name = st.text_input("👤 Tên khách hàng hoặc Mã thẻ:", "Khách hàng thân thiết")

    # 3. XỬ LÝ LOGIC DỰ BÁO & TIỀN XỬ LÝ
    st.markdown("### 🚀 Kết quả phân tích hành vi")
    
    if st.button("Thực hiện dự báo ngay"):
        with st.spinner('Mô hình đang tính toán...'):
            # 1. Lọc luật dựa trên sản phẩm đã chọn
            input_set = {selected_item}
            res = model_rules[model_rules['antecedents'].apply(lambda x: input_set.issubset(x))]
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
# --- ĐÁNH GIÁ & HIỆU NĂNG (EVALUATION) ---
elif page == " Đánh giá & Hiệu năng":
    st.title("📊 Đánh giá & Hiệu năng Mô hình")
    st.markdown("---")

    # 1. CÁC CHỈ SỐ ĐO LƯỜNG ĐẶC THÙ 
    st.subheader("1. Các chỉ số đo lường chất lượng luật")
    
    try:
        with open('models/file.pkl', 'rb') as f:
            rules_eval = pickle.load(f)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Tổng số luật tìm được", f"{len(rules_eval)}")
        col_m2.metric("Độ tin cậy TB (Confidence)", f"{rules_eval['confidence'].mean():.2%}")
        col_m3.metric("Độ tương quan TB (Lift)", f"{rules_eval['lift'].mean():.2f}")
        c4, c5 = st.columns(2)
        col_m1.metric("Độ tin cậy cao nhất", f"{rules_eval['confidence'].max():.2%}")
        col_m2.metric("Số luật có Lift > 2.0", f"{len(rules_eval[rules_eval['lift'] > 2])}")
        st.info("""
        **Giải thích chỉ số:**
        - **Support (Độ hỗ trợ):** Tần suất xuất hiện của cặp sản phẩm trong toàn bộ dữ liệu.
        - **Confidence (Độ tin cậy):** Xác suất khách mua món B khi đã cầm món A trên tay.
        - **Lift (Độ tương quan):** Chỉ số quan trọng nhất. Lift > 1 chứng tỏ mối quan hệ là thực tế (không phải ngẫu nhiên).
        """)

        # 2. BIỂU ĐỒ KỸ THUẬT: Scatter Plot (Thay cho Confusion Matrix)
        st.subheader("2. Biểu đồ kỹ thuật: Phân tán luật kết hợp")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(rules_eval['support'], rules_eval['confidence'], 
                           c=rules_eval['lift'], cmap='YlOrRd', alpha=0.6)
        plt.colorbar(scatter, label='Chỉ số Lift')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Tương quan giữa Support, Confidence và Lift')
        st.pyplot(fig)

        st.subheader("3. Heatmap: Mối tương quan giữa các cặp sản phẩm (Top Lift)")
        # Lấy top 20 luật có Lift cao nhất để Heatmap không bị quá dày đặc
        top_rules = rules_eval.sort_values(by='lift', ascending=False).head(20).copy()
        # Chuyển đổi frozenset sang string để hiển thị trên trục biểu đồ
        top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        # Tạo bảng pivot cho Heatmap
        pivot_table = top_rules.pivot(index='antecedents_str', columns='consequents_str', values='lift')
        # Vẽ biểu đồ
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax3)
        ax3.set_title('Ma trận tương quan Lift giữa các nhóm sản phẩm')
        ax3.set_xlabel('Sản phẩm gợi ý (Consequents)')
        ax3.set_ylabel('Sản phẩm khách mua (Antecedents)')

        st.pyplot(fig3)
        st.subheader("4. Phân tích sai số và Nhận định")
        # 3. PHÂN TÍCH SAI SỐ & HẠN CHẾ
        with st.expander("🔍 Tại sao mô hình có thể dự báo chưa tối ưu?", expanded=True):
            st.write("""
            - **Trường hợp sai số:** Mô hình thường đưa ra các gợi ý có Confidence thấp (dưới 20%). 
            - **Nguyên nhân:** Do dữ liệu siêu thị cực kỳ phân tán. Một khách mua sữa có thể chọn kèm theo 160 loại mặt hàng khác nhau, dẫn đến xác suất cho một món cụ thể bị chia nhỏ.
            - **Lệch dữ liệu:** Các mặt hàng bán quá chạy (Whole milk) xuất hiện trong hầu hết các luật, làm lu mờ các quy luật của những mặt hàng ngách.
            """)

        # 4. HƯỚNG CẢI THIỆN
        st.subheader("🚀 Hướng cải thiện")
        st.success("""
        1. **Phân đoạn khách hàng (Clustering):** Chia khách hàng thành các nhóm (Ví dụ: Nhóm thích đồ ngọt, nhóm nội trợ) trước khi chạy FP-Growth để có luật chính xác hơn cho từng nhóm.
        2. **Bổ sung biến thời gian:** Phân tích theo mùa (Tết mua gì, Hè mua gì) để gợi ý mang tính thời điểm cao hơn.
        3. **Dữ liệu lớn hơn:** Thu thập thêm lịch sử giao dịch để tăng độ tin cậy cho các mặt hàng ít phổ biến.
        """)

    except FileNotFoundError:
        st.error("⚠️ Vui lòng chạy huấn luyện mô hình để có dữ liệu đánh giá!")
