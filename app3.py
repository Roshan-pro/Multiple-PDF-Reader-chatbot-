import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
import io

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract images from PDFs using PyMuPDF (fitz)
def extract_images(pdf_docs):
    images = []
    for pdf in pdf_docs:
        # Open the file-like object instead of file path
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

# Function to generate embeddings for text chunks
def generate_embeddings(chunks):
    return model.encode(chunks)

# Function to perform similarity search in the FAISS index
def perform_similarity_search(query, faiss_index, text_chunks):
    query_embedding = model.encode([query])  # Embed the query
    D, I = faiss_index.search(np.array(query_embedding).astype(np.float32), k=3)  # Search top 3 closest chunks
    return [text_chunks[i] for i in I[0]]

# Function to clean and format the text answer
def clean_answer(text):
    # Remove excessive line breaks, unwanted spaces, and correct broken words
    text = text.replace('\n', ' ').replace('  ', ' ').strip()
    # Optional: You can also apply more advanced text cleaning techniques here if necessary
    return text

# Function to get the answer based on user question
def get_answer(user_question, faiss_index, text_chunks):
    relevant_chunks = perform_similarity_search(user_question, faiss_index, text_chunks)
    answer = " ".join(relevant_chunks)
    cleaned_answer = clean_answer(answer)
    return cleaned_answer

# Main function to handle the Streamlit app
def main():
    st.set_page_config('PDF Chat Bot')
    st.title("PDF Reader Chatbot")

    # Initialize faiss_index, text_chunks, and images to be used globally
    faiss_index = None
    text_chunks = None
    images = []

    # User input for the question
    user_question = st.text_input("Ask a question based on the uploaded PDF:")

    # Sidebar for file upload
    with st.sidebar:
        st.title('Menu:')
        pdf_docs = st.file_uploader("Upload your PDF Files", type='pdf', accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract raw text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Extract images from the PDFs
                images = extract_images(pdf_docs)
                
                # Split text into manageable chunks
                text_chunks = split_text(raw_text)
                
                # Generate embeddings for the text chunks
                embeddings = generate_embeddings(text_chunks)
                
                # Create a FAISS index for the embeddings
                dimension = embeddings.shape[1]  # Get the embedding dimension
                faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
                faiss_index.add(np.array(embeddings).astype(np.float32))  # Add the embeddings to the FAISS index
                
                st.success("PDF Processing is done!")

    # If the user asks a question and the index is ready
    if user_question:
        if faiss_index is not None and text_chunks is not None:
            # Get the answer from the PDF content based on the user's question
            answer = get_answer(user_question, faiss_index, text_chunks)
            
            # Display the answer in the main area with better formatting
            st.markdown("### Answer:")
            st.write(answer)  # Display the cleaned and formatted answer

            # Display related images (if any) from the PDF
            if images:
                st.markdown("### Related Images:")
                for img in images:
                    st.image(img, use_column_width=True)
            else:
                st.write("No images found in the PDF.")
        else:
            st.error("Please upload and process the PDF first before asking a question.")

# This will run the Streamlit app
if __name__ == "__main__":
    main()
