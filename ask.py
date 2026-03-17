from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("--- טוען מערכת (Manual RAG Mode) ---")

# 1. טעינת המודלים הבסיסיים (כבר בדקנו שהם עובדים)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = OllamaLLM(model="phi3")

print("\n--- הצ'אט מוכן! ---")

while True:
    query = input("\nשאל שאלה על המסמך (או 'exit' ליציאה): ")
    if query.lower() == 'exit':
        break
    
    print("מחפש במידע שלך...")
    
    # א. חיפוש ידני של המקטעים הכי רלוונטיים (Top 3)
    docs = vectorstore.similarity_search(query, k=3)
    
    # ב. איחוד המקטעים לטקסט אחד (Context)
    context = "\n\n".join([d.page_content for d in docs])
    
    # ג. בניית ה-Prompt בעצמנו
    full_prompt = f"""
    You are a helpful assistant. Use the context below to answer the user's question.
    If the answer is not in the context, say you don't know.
    
    Context:
    {context}
    
    User Question: {query}
    
    Answer:"""
    
    print("ה-AI מנסח תשובה...")
    
    # ד. שליחה ישירה ל-AI
    response = llm.invoke(full_prompt)
    
    print("\nתשובה:")
    print(response)