import streamlit as st
from PyPDF2 import PdfReader
import fitz  
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
import io

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def extract_images(pdf_docs):
    images = []
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
    return images

# Function to split text into manageable chunks
def split_text(text, chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks

def generate_embeddings(chunks):
    return model.encode(chunks)

def perform_similarity_search(query, faiss_index, text_chunks):
    query_embedding = model.encode([query])  
    D, I = faiss_index.search(np.array(query_embedding).astype(np.float32), k=3)  
    return [text_chunks[i] for i in I[0]]

def clean_answer(text):
    text = text.replace('\n', ' ').replace('  ', ' ').strip()
    return text

def get_answer(user_question, faiss_index, text_chunks):
    relevant_chunks = perform_similarity_search(user_question, faiss_index, text_chunks)
    answer = " ".join(relevant_chunks)
    cleaned_answer = clean_answer(answer)
    return cleaned_answer

def main():
    st.set_page_config('PDF Chat Bot')
    st.title("PDF Reader Chatbot")

    faiss_index = None
    text_chunks = None
    images = []

    user_question = st.text_input("Ask a question based on the uploaded PDF:")

    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader("Upload your PDF Files", type='pdf', accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                images = extract_images(pdf_docs)
                
                text_chunks = split_text(raw_text)
                
                embeddings = generate_embeddings(text_chunks)
                
                dimension = embeddings.shape[1]  
                faiss_index = faiss.IndexFlatL2(dimension)  
                faiss_index.add(np.array(embeddings).astype(np.float32)) 
                
                st.success("PDF Processing is done!")

    if user_question:
        if faiss_index is not None and text_chunks is not None:
            answer = get_answer(user_question, faiss_index, text_chunks)
            
            st.markdown("### Answer:")
            st.write(answer)  

            if images:
                st.markdown("### Related Images:")
                for img in images:
                    st.image(img, use_column_width=True)
            else:
                st.write("No images found in the PDF.")
        else:
            st.error("Please upload and process the PDF first before asking a question.")

if __name__ == "__main__":
    main()
