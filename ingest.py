# יבוא הכלים שנצרכים
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. טעינת ה-PDF
# אנחנו משתמשים ב-PyPDFLoader שיודע לפרק את המבנה של קובץ PDF לטקסט נקי
print("--- שלב 1: טוען את ה-PDF ---")
loader = PyPDFLoader("my_document.pdf")
docs = loader.load()

# 2. פיצול הטקסט למקטעים (Chunks)
# chunk_size=1000 אומר שכל חתיכה תהיה בערך 1000 תווים
# chunk_overlap=200 אומר שכל חתיכה תכיל קצת מהסוף של החתיכה הקודמת (כדי לא לאבד הקשר)
print("--- שלב 2: מפצל את הטקסט למקטעים ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. הגדרת המודל שהופך טקסט למספרים (Embeddings)
# אנחנו משתמשים במודל קטן וחינמי של HuggingFace שרץ אצלך מקומית
print("--- שלב 3: יוצר Embeddings (זה עשוי לקחת דקה בפעם הראשונה) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. שמירה בבסיס הנתונים הוקטורי (ChromaDB)
# המידע יישמר בתיקייה בשם 'db' בתוך הפרויקט שלך
print("--- שלב 4: שומר את הנתונים ב-ChromaDB ---")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("--- הסתיים בהצלחה! הזיכרון מוכן ---")