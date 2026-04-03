import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import streamlit as st  # <-- Thêm thư viện này

# --- GIAO DIỆN WEB ---
st.title("🛒 Ứng dụng Phân tích Giỏ hàng")
st.write("Đang xử lý dữ liệu, vui lòng đợi trong giây lát...")

# 1. Đọc dữ liệu
# Lưu ý: Đảm bảo đường dẫn này đúng với cấu trúc trên GitHub của bạn
data = pd.read_csv('data/Groceries_dataset.csv')

# 2. TIỀN XỬ LÝ
transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()

# 3. VECTOR HÓA
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 4. CHẠY THUẬT TOÁN APRIORI
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)

# 5. TRÍCH XUẤT LUẬT KẾT HỢP
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules[rules['confidence'] > 0.05]

# --- 6. HIỂN THỊ KẾT QUẢ LÊN STREAMLIT ---
st.subheader("Top 10 luật kết hợp mạnh nhất:")

# Lấy 10 luật mạnh nhất
top_10 = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10)

# Mẹo nhỏ: Streamlit hiển thị frozenset (dạng mẫu) hơi xấu, 
# nên chúng ta chuyển nó sang dạng list/string cho đẹp
top_10['antecedents'] = top_10['antecedents'].apply(lambda x: ', '.join(list(x)))
top_10['consequents'] = top_10['consequents'].apply(lambda x: ', '.join(list(x)))

# Dùng st.dataframe để hiện bảng có thể cuộn và phóng to
st.dataframe(top_10)

st.success("Đã tải xong kết quả!")
