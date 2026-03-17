import streamlit as st
import requests

# 1. הגדרות דף
st.set_page_config(page_title="Personal AI Search", page_icon="🔍")
st.title("🔍 מנוע חיפוש אישי מבוסס AI")

# סרגל צד להעלאת קבצים
with st.sidebar:
    st.header("העלאת מסמכים")
    uploaded_file = st.file_uploader("בחר קובץ PDF", type="pdf")
    
    if st.button("עבד קובץ"):
        if uploaded_file is not None:
            with st.spinner("מעבד את הקובץ..."):
                # שליחת הקובץ ל-API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post("http://127.0.0.1:8000/ingest", files=files)
                
                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error("שגיאה בעיבוד הקובץ.")
        else:
            st.warning("אנא בחר קובץ קודם.")

# כתובת השרת שבנינו קודם
API_URL = "http://127.0.0.1:8000/ask"

# 2. ניהול היסטוריית הצ'אט
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת הודעות קודמות
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. תיבת הקלט מהמשתמש
if prompt := st.chat_input("שאל אותי משהו על המסמכים שלך..."):
    # הצגת הודעת המשתמש
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # שליחת בקשה ל-API שלנו
    with st.chat_message("assistant"):
        with st.spinner("ה-AI מחפש במסמכים..."):
            try:
                response = requests.post(API_URL, json={"question": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("---")
                        st.caption("📍 המידע התבסס על המקורות הבאים:")
                        for s in sources:
                            # ניקוי שם הקובץ מהנתיב המלא (אם קיים)
                            clean_source = s.split("\\")[-1].split("/")[-1]
                            st.info(f"📄 {clean_source}")
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("שגיאה בתקשורת עם השרת.")
            except Exception as e:
                st.error(f"לא ניתן להתחבר ל-API: {e}")