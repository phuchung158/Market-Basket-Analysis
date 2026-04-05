import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
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
    Từ đó có thể chủ động gợi ý sản phẩm đi kèm. Khi khách hàng mua món A, hệ thống sẽ tự động "nhắc" món B, từ đó tăng kích thước giỏ hàng và doanh thu trên mỗi hóa đơn.""")

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
# 1. Cấu hình CSS (Giữ nguyên phong cách Food Association)
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .result-card {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
        border-top: 5px solid #4B0082;
    }
    .badge {
        background-color: #4B0082; color: white; padding: 5px 12px;
        border-radius: 50%; font-weight: bold; margin-right: 10px;
    }
    .food-title { font-size: 20px; font-weight: bold; color: #333; display: inline; }
    .metric-box {
        background-color: #f8f9fa; padding: 10px; border-radius: 8px;
        text-align: center; border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# 2. GỌI MÔ HÌNH (Load model_rules từ file .pkl)
try:
    with open('models/file.pkl', 'rb') as f:
        model_rules = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Không tìm thấy file models/file.pkl. Hãy huấn luyện mô hình trước!")
    st.stop()

# 3. SIDEBAR: Nhập liệu
with st.sidebar:
    st.title("🍴 Food Association")
    st.subheader("🔍 Tìm Kiếm")
    
    # Lấy danh sách sản phẩm từ antecedents trong mô hình
    all_products = sorted(set([list(x)[0] for x in model_rules['antecedents']]))
    selected_item = st.selectbox("Chọn món ăn:", all_products)
    
    num_rec = st.select_slider("Số lượng gợi ý:", options=[3, 5, 8, 10], value=5)
    conf_threshold = st.slider("Ngưỡng tin cậy:", 0.01, 0.5, 0.05)
    
    predict_btn = st.button("🔍 Tìm gợi ý", use_container_width=True)

# 4. NỘI DUNG CHÍNH: Xử lý logic gọi mô hình
if predict_btn:
    st.header("Kết Quả Gợi Ý")
    
    # Logic lọc mô hình: Tìm các luật chứa sản phẩm đã chọn
    input_set = {selected_item}
    res = model_rules[model_rules['antecedents'].apply(lambda x: input_set.issubset(x))]
    res = res[res['confidence'] >= conf_threshold]
    
    # Tiền xử lý kết quả
    res['suggested_name'] = res['consequents'].apply(lambda x: list(x)[0])
    results = res.sort_values(by='lift', ascending=False).drop_duplicates(subset=['suggested_name']).head(num_rec)

    if not results.empty:
        st.caption(f"Tìm thấy {len(results)} gợi ý cho '{selected_item}'")
        
        # Chia cột để hiển thị Card (2 cột mỗi hàng)
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(results):
                    row = results.iloc[i + j]
                    name = row['suggested_name']
                    conf = row['confidence']
                    lift = row['lift']
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class="result-card">
                            <span class="badge">{i + j + 1}</span> <p class="food-title">{name}</p>
                            <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                                <div class="metric-box" style="width: 48%;">
                                    <p style="font-size: 11px; color: #666; margin: 0;">📉 Độ tin cậy</p>
                                    <p style="font-size: 16px; color: #28a745; font-weight: bold; margin: 0;">{conf:.1%}</p>
                                </div>
                                <div class="metric-box" style="width: 48%;">
                                    <p style="font-size: 11px; color: #666; margin: 0;">🚀 Liên quan (Lift)</p>
                                    <p style="font-size: 16px; color: #007bff; font-weight: bold; margin: 0;">{lift:.2f}x</p>
                                </div>
                            </div>
                            <p style="font-size: 12px; color: #555; margin-top: 15px; padding: 8px; background: #f1f3f5; border-radius: 5px;">
                                Khách mua <b>{selected_item}</b> có <b>{conf:.1%}</b> khả năng mua thêm <b>{name}</b>.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.warning(f"Không tìm thấy gợi ý nào cho '{selected_item}' với ngưỡng tin cậy này.")
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
