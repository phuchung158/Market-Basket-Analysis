🛒 Market Basket Analysis - Tối ưu hóa Giỏ hàng với Thuật toán Apriori

. Ứng dụng khai phá luật kết hợp từ dữ liệu giao dịch siêu thị để đưa ra các chiến lược gợi ý sản phẩm thông minh.

## 📝 Giới thiệu bài toán
* **Đối tượng:** Dữ liệu giao dịch mua sắm (Groceries Dataset).
* **Mục tiêu:** Tìm ra các quy luật mua sắm đồng thời của khách hàng (Ví dụ: Khách mua Sữa thường mua kèm Xúc xích).
* **Giá trị:** Hỗ trợ doanh nghiệp sắp xếp kệ hàng, tạo combo khuyến mãi và xây dựng hệ thống gợi ý (Recommendation System).

## 🛠 Công nghệ sử dụng
* **Ngôn ngữ:** Python 3.x
* **Thư viện chính:** * `mlxtend`: Triển khai thuật toán Apriori và Association Rules.
    * `pandas`: Tiền xử lý và làm sạch dữ liệu.
    * `streamlit`: Xây dựng giao diện ứng dụng Web trực quan.
    * `matplotlib` & `seaborn`: Trực quan hóa các chỉ số đo lường.

## 📂 Cấu trúc thư mục
```text
├── app.py              # File chạy chính (Streamlit App)
├── requirements.txt    # Danh sách thư viện cần cài đặt
├── data/               # Thư mục chứa dữ liệu
│   └── Groceries_dataset.csv
└── README.md           # Hướng dẫn dự án