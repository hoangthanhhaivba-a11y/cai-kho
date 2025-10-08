# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import os # Vẫn giữ lại để kiểm tra biến môi trường cho running local

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- KHỞI TẠO VÀ CẤU HÌNH GEMINI ---
# Lấy Khóa API: Ưu tiên st.secrets (cho Cloud), sau đó là os.environ (cho Local)
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=API_KEY)
        MODEL_NAME = 'gemini-2.5-flash'
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        GEMINI_CLIENT = None
else:
    GEMINI_CLIENT = None

# 1. Khởi tạo State cho Chat
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
    
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét Tóm tắt (Chức năng 5) ---
# Hàm này tách biệt với khung chat (Chức năng 6)
def get_ai_analysis(data_for_ai):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not GEMINI_CLIENT:
        return "Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình 'GEMINI_API_KEY'."

    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
    
    Dữ liệu thô và chỉ số:
    {data_for_ai}
    """

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}. Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # 2. Tái tạo Phiên Chat (Quan trọng để cập nhật ngữ cảnh)
            processed_data_markdown = df_processed.to_markdown(index=False)
            
            if GEMINI_CLIENT:
                try:
                    # Gán dữ liệu đã xử lý vào System Instruction để làm ngữ cảnh
                    system_instruction = (
                        "Bạn là một Trợ lý AI chuyên nghiệp về Phân tích Tài chính. "
                        "Hãy trả lời các câu hỏi dựa trên dữ liệu Báo cáo Tài chính đã xử lý mà tôi cung cấp. "
                        "Dữ liệu phân tích đã được tải:\n\n"
                        f"{processed_data_markdown}"
                    )
                    
                    # Tạo phiên chat mới
                    st.session_state["chat_session"] = GEMINI_CLIENT.chats.create(
                        model=MODEL_NAME,
                        system_instruction=system_instruction
                    )
                    
                    # Tin nhắn chào mừng ban đầu
                    welcome_message = "Dữ liệu Báo cáo Tài chính đã được tải và xử lý. Bạn có thể hỏi tôi chi tiết về các chỉ tiêu tăng trưởng, tỷ trọng hoặc khả năng thanh toán ngay bây giờ!"
                    st.session_state["chat_messages"] = [{"role": "assistant", "content": welcome_message}]
                    
                except Exception as e:
                    st.error(f"Không thể khởi tạo phiên chat: {e}")
                    st.session_state["chat_session"] = None


            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width
