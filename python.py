import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# Khởi tạo khóa API và Client (Chỉ chạy một lần)
# Lấy API Key từ Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
else:
    # Khởi tạo Client toàn cục
    try:
        gemini_client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo Gemini Client: {e}")


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
        # Nếu không tìm thấy, cố gắng tìm các tên gọi khác (ví dụ: Tổng cộng tài sản, TỔNG TÀI SẢN)
        tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('tài sản|tổng cộng', case=False, na=False)]
        if tong_tai_san_row.empty:
             raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc chỉ tiêu tương đương.")

    # Lấy giá trị của chỉ tiêu Tổng Tài sản
    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho phân tích tự động (Chức năng 5) ---
def get_ai_analysis(data_for_ai):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not API_KEY:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."
        
    try:
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Xử lý Chat hỏi đáp với Gemini ---
def get_gemini_chat_response(messages, system_prompt):
    """
    Gửi lịch sử trò chuyện và system prompt đến Gemini API.
    Messages là list các dict có dạng {"role": "user"|"assistant", "content": "..."}
    """
    if not API_KEY:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."

    # Chuyển đổi định dạng tin nhắn của Streamlit sang định dạng của Gemini API
    # Role 'assistant' trong Streamlit tương ứng với 'model' trong Gemini API
    contents = []
    for message in messages:
        role = 'model' if message["role"] == 'assistant' else 'user'
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            system_instruction=system_prompt,
        )
        return response.text
    except Exception as e:
        return f"Lỗi Gemini: {e}"

# --- Hàm Chat Interface ---
def run_gemini_chat_interface(df_processed):
    
    # 1. Khởi tạo System Prompt (Gán dữ liệu phân tích)
    system_instruction = (
        "Bạn là một trợ lý phân tích tài chính chuyên nghiệp. "
        "Các câu trả lời của bạn phải dựa trên dữ liệu báo cáo tài chính sau (được định dạng Markdown):"
        f"\n\n{df_processed.to_markdown(index=False)}\n\n"
        "Hãy trả lời các câu hỏi của người dùng về dữ liệu này bằng tiếng Việt. "
        "Nếu người dùng hỏi một câu hỏi không liên quan đến dữ liệu, hãy lịch sự từ chối và yêu cầu họ hỏi về báo cáo tài chính."
    )
    
    # 2. Khởi tạo Lịch sử trò chuyện (Nếu chưa có)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI Phân tích Tài chính. Bạn có câu hỏi nào về báo cáo tài chính đã tải lên không?"}
        ]
        
    # 3. Hiển thị lịch sử tin nhắn
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Xử lý input từ người dùng
    if prompt := st.chat_input("Hỏi Gemini về dữ liệu đã tải lên..."):
        
        # Thêm tin nhắn người dùng vào lịch sử và hiển thị
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Lấy phản hồi từ Gemini
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý, Gemini đang phân tích dữ liệu..."):
                
                # Gọi API với toàn bộ lịch sử trò chuyện và System Prompt mới (có chứa dữ liệu)
                ai_response = get_gemini_chat_response(
                    st.session_state["chat_messages"], 
                    system_instruction
                )
                
                st.markdown(ai_response)
                # Lưu phản hồi của AI
                st.session_state["chat_messages"].append({"role": "assistant", "content": ai_response})


# --- CHỨC NĂNG CHÍNH CỦA APP ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        # Xử lý file
        df_raw = pd.read_excel(uploaded_file)
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N = "N/A" 
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                # **LƯU Ý: Đảm bảo file Excel của bạn có chỉ tiêu Nợ Ngắn Hạn**
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, tránh chia cho 0
                if no_ngan_han_N != 0:
                     thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                # Hiển thị Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N
                    )
                with col2:
                    delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N,
                        delta=f"{delta_value:.2f}" if delta_value is not None else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except Exception as e:
                 st.error(f"Lỗi khi tính chỉ số tài chính: {e}")
            
            
            # --- Chức năng 5: Nhận xét AI Tự động ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tự động)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Tái sử dụng logic cũ)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else str(thanh_toan_hien_hanh_N_1),
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else str(thanh_toan_hien_hanh_N)
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích Tự động"):
                if API_KEY:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai)
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API để thực hiện phân tích.")

            st.markdown("---")
            
            # --- CHỨC NĂNG MỚI: KHUNG CHAT HỎI ĐÁP ---
            with st.expander("💬 6. Chat hỏi đáp với Gemini về Dữ liệu này"):
                if API_KEY:
                    run_gemini_chat_interface(df_processed)
                else:
                    st.warning("Không thể khởi tạo khung chat do thiếu Khóa API Gemini.")
            
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        # Reset chat state khi lỗi cấu trúc
        if "chat_messages" in st.session_state:
            del st.session_state["chat_messages"]
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
        if "chat_messages" in st.session_state:
            del st.session_state["chat_messages"]
            
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    # Reset chat state khi không có file
    if "chat_messages" in st.session_state:
        del st.session_state["chat_messages"]
