from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import UploadFile, File
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. הגדרת האפליקציה
app = FastAPI(title="Personal Files AI Search Engine")

# 2. טעינת המודלים (מחוץ לנקודות הקצה כדי שלא ייטענו מחדש בכל בקשה)
print("--- טוען מודלים לזיכרון ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = OllamaLLM(model="phi3")

# 3. הגדרת מבנה הבקשה (מראה על מקצועיות ושימוש ב-Pydantic)
class QueryRequest(BaseModel):
    question: str

# 4. נקודת קצה לבדיקה (Health Check)
@app.get("/")
def home():
    return {"message": "AI Search Engine is UP and running!"}

# 5. נקודת הקצה העיקרית - החיפוש
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # חיפוש ב-ChromaDB
        docs = vectorstore.similarity_search(request.question, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        # בניית ה-Prompt
        full_prompt = f"""
        Use the context below to answer the user question.
        Context: {context}
        User Question: {request.question}
        Answer:"""
        
        # הרצה
        response = llm.invoke(full_prompt)
        
        # החזרת התשובה פלוס המקורות (Citations) - זה מרשים מראיינים!
        sources = [d.metadata.get("source", "Unknown") for d in docs]
        
        return {
            "answer": response,
            "sources": list(set(sources)) # מסיר כפילויות
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    try:
        # 1. שמירת הקובץ זמנית בשרת
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 2. תהליך ה-Ingestion (בדיוק כמו ב-ingest.py)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # הוספת המקטעים החדשים לבסיס הנתונים הקיים
        vectorstore.add_documents(documents=splits)
        
        # 3. מחיקת הקובץ הזמני
        os.remove(file_path)
        
        return {"message": f"הקובץ {file.filename} עובד ונוסף לזיכרון בהצלחה!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    