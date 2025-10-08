# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import os 

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
            
            # **Đoạn mã đã sửa lỗi cú pháp:**
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True) # <--- Đã đóng ngoặc đơn ')'
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"

            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "∞",
                    )
                with col2:
                    delta_value = "N/A"
                    if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf'):
                         delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                         
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "∞",
                        delta=delta_value
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                 st.warning("Nợ Ngắn Hạn bằng 0. Chỉ số thanh toán là Vô cực (rất tốt).")
            
            # --- Chức năng 5: Nhận xét AI (Giữ nguyên logic) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    processed_data_markdown,
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                if GEMINI_CLIENT:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY'.")


            # -----------------------------------------------------------------
            # --- CHỨC NĂNG 6: KHUNG CHAT HỎI ĐÁP CHUYÊN SÂU VỚI GEMINI 💬 ---
            # -----------------------------------------------------------------
            st.subheader("6. Hỏi đáp Chuyên sâu với Gemini 💬")
            
            if st.session_state["chat_session"]:
                
                # Hiển thị lịch sử chat
                for message in st.session_state["chat_messages"]:
                    avatar = "🤖" if message["role"] == "assistant" else "👤"
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(message["content"])

                # Xử lý đầu vào người dùng
                if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính này..."):
                    
                    # Thêm tin nhắn người dùng vào lịch sử
                    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
                    
                    # Hiển thị tin nhắn người dùng
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)

                    try:
                        # Gọi API Gemini (Sử dụng session chat đã có ngữ cảnh)
                        with st.spinner("Đang chờ phản hồi từ chuyên gia AI..."):
                            response = st.session_state["chat_session"].send_message(prompt)
                            full_response = response.text
                        
                        # Hiển thị phản hồi AI
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(full_response)
                        
                        # Thêm phản hồi AI vào lịch sử chat
                        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})

                    except APIError as e:
                        error_message = f"Lỗi API Gemini: {e.message}. Vui lòng kiểm tra API Key."
                        st.error(error_message)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": error_message})
                        
                    except Exception as e:
                        error_message = f"Đã xảy ra lỗi không xác định: {e}"
                        st.error(error_message)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": error_message})
            else:
                st.info("Vui lòng tải file để khởi tạo phiên chat hoặc kiểm tra Khóa API Gemini.")
            # -----------------------------------------------------------------

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
