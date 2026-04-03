import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. Đọc dữ liệu (Thay đường dẫn file của bạn vào đây)
data = pd.read_csv(r'D:\PhucHung\Apriori\data\Groceries_dataset')

# 2. TIỀN XỬ LÝ (LÀM SẠCH)
# Gom các sản phẩm của cùng 1 người mua trong cùng 1 ngày thành 1 giỏ hàng
transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).values.tolist()

# 3. VECTOR HÓA (ONE-HOT ENCODING)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 4. CHẠY THUẬT TOÁN APRIORI
# Tìm tập mục phổ biến với Support >= 0.01
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)

# 5. TRÍCH XUẤT LUẬT KẾT HỢP
# Tìm các luật có Lift > 1
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules[rules['confidence'] > 0.05]
# 6. IN KẾT QUẢ RA MÀN HÌNH
print("Top 10 luật kết hợp mạnh nhất:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))