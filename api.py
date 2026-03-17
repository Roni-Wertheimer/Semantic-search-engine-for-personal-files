from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import UploadFile, File
import os
from typing import List
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
        # 1. חיפוש המקטעים הרלוונטיים
        docs = vectorstore.similarity_search(request.question, k=3)
        
        # 2. איסוף המקורות (שם קובץ + עמוד)
        sources_info = []
        for d in docs:
            file_name = d.metadata.get("source", "Unknown File")
            # ה-Loader מתחיל לספור עמודים מ-0, אז נוסיף 1 כדי שיהיה ברור למשתמש
            page_num = d.metadata.get("page", 0) + 1
            sources_info.append(f"{file_name} (עמוד {page_num})")
        
        # 3. בניית ההקשר ל-AI
        context = "\n\n".join([d.page_content for d in docs])
        full_prompt = f"Context: {context}\n\nQuestion: {request.question}\nAnswer:"
        
        # 4. הרצה
        response = llm.invoke(full_prompt)
        
        return {
            "answer": response,
            "sources": list(set(sources_info))  # הסרת מקורות כפולים
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    processed_files = []
    try:
        global vectorstore
        for file in files:
            # 1. שמירה זמנית
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # 2. עיבוד הקובץ
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # 3. הוספה ל-Vectorstore
            vectorstore.add_documents(documents=splits)
            
            # 4. מחיקת הקובץ הזמני
            os.remove(file_path)
            processed_files.append(file.filename)
            
        return {"message": f"הקבצים הבאים נוספו בהצלחה: {', '.join(processed_files)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 1. נקודת קצה לקבלת רשימת הקבצים הקיימים בזיכרון
@app.get("/list_files")
async def list_files():
    try:
        data = vectorstore.get()
        metadatas = data.get("metadatas", [])
        
        unique_files = []
        seen_internal_names = set()
        
        for m in metadatas:
            internal_name = m.get("source")
            if internal_name and internal_name not in seen_internal_names:
                display_name = internal_name.split("\\")[-1].split("/")[-1]
                # אנחנו מחזירים אובייקט עם שני השמות
                unique_files.append({
                    "display_name": display_name,
                    "internal_name": internal_name
                })
                seen_internal_names.add(internal_name)
        
        return {"files": unique_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_file")
async def delete_file(internal_name: str):
    try:
        print(f"--- מנסה למחוק את: {internal_name} ---")
        
        # מחיקה עם התאמה מדויקת למקור
        vectorstore.delete(where={"source": internal_name})
        
        print(f"--- המחיקה הסתיימה בהצלחה ---")
        return {"message": "Success"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reset")
async def reset_database():
    try:
        # חייב להופיע כאן, לפני כל שימוש במשתנה!
        global vectorstore 
        
        # עכשיו אפשר להשתמש בו ולשנות אותו
        vectorstore.delete_collection()
        
        # יצירה מחדש
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        return {"message": "הזיכרון אופס בהצלחה! המערכת נקייה."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   