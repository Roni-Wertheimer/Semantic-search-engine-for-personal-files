import langchain
import chromadb

from langchain_ollama import OllamaLLM

# 1. הגדרת המודל - אנחנו אומרים ללנגצ'יין לדבר עם Ollama שרץ אצלך
# ומשתמשים ב-phi3 כי הוא קל ומהיר
llm = OllamaLLM(model="phi3")

# 2. שליחת שאלה לבדיקה
question = "מהם שלושת השלבים העיקריים בבניית מערכת RAG?"
response = llm.invoke(question)

print("--- תשובת המודל ---")
print(response)