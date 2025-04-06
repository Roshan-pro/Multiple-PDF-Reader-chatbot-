# Multiple-PDF-Reader-chatbot-
PDF Reader Chatbot
## Overview
The PDF Reader Chatbot is a web-based application built with Streamlit, PyPDF2, FAISS, and Sentence-Transformers. It allows users to upload PDF files, processes them to extract both text and images, and enables users to ask questions based on the contents of the uploaded PDF files.

The chatbot answers user questions by performing a semantic similarity search on the text extracted from the PDF documents. It provides highly relevant responses by searching for the most similar sections in the document to the user’s query. If applicable, it also displays any related images from the PDF files that are relevant to the question asked.

## Key Features
Upload PDF Files: Users can upload one or more PDF files through the web interface.

Text Extraction: Extracts text from uploaded PDF documents using PyPDF2.

Image Extraction: Extracts images from PDFs and displays them alongside answers (if available).

Semantic Search: Uses Sentence-Transformers and FAISS for efficient similarity search to provide relevant answers based on the content of the PDFs.

Answer Display: The chatbot shows structured, readable answers to the user's questions based on the content of the PDFs.

Real-time Processing: Users can ask questions and get answers instantly after uploading the PDF files.

## Technologies Used
Streamlit: For building the interactive web interface.

PyPDF2: To extract text from PDF files.

PyMuPDF (fitz): To extract images from PDF files.

FAISS: For creating a fast similarity search index.

Sentence-Transformers: For generating text embeddings and performing semantic similarity searches.
