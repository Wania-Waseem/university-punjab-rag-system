# =====================================================
# STREAMLIT WEB INTERFACE FOR RAG SYSTEM WITH PDF UPLOAD
# Save this as: streamlit_app.py
# =====================================================

import streamlit as st
import os
import requests
import PyPDF2
from io import BytesIO
import numpy as np
from typing import List, Dict
import time
from datetime import datetime
import tempfile
import shutil

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

# Set page configuration
st.set_page_config(
    page_title="University of Punjab Admission Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left-color: #4caf50;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'document_text' not in st.session_state:
    st.session_state.document_text = ""
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = ""

class PDFProcessor:
    """Handle PDF file processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def process_pdf_file(uploaded_file) -> tuple:
        """Process uploaded PDF and return text content"""
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text
                with open(tmp_file_path, 'rb') as file:
                    text = PDFProcessor.extract_text_from_pdf(file)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                return text, True
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return "", False
        
        return "", False

class SimpleRAGSystem:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def query(self, question: str) -> str:
        """Enhanced RAG query function"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return "❌ I couldn't find relevant information in the uploaded document. Please make sure the PDF contains information related to your question."
            
            # Combine context from retrieved documents
            context_parts = []
            sources = set()
            
            for doc in relevant_docs:
                context_parts.append(doc.page_content)
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            context = "\n\n".join(context_parts)
            
            # Create comprehensive response
            response = "📄 **Based on the uploaded university guidelines:**\n\n"
            
            # Add most relevant context
            if context:
                # Limit context length for readability
                max_context_length = 800
                truncated_context = context[:max_context_length]
                if len(context) > max_context_length:
                    truncated_context += "...\n\n[Additional relevant information found in document]"
                
                response += f"**Relevant Information:**\n{truncated_context}\n\n"
            
            # Add contextual answer
            response += "💡 **Direct Answer:**\n"
            
            # Simple keyword-based enhancement
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Look for specific information in the context
            if any(word in context_lower for word in ['eligibility', 'criteria', 'requirement']):
                response += "The document contains specific eligibility criteria and requirements. "
            
            if any(word in context_lower for word in ['fee', 'cost', 'payment']):
                response += "Fee and cost information is available in the document. "
            
            if any(word in context_lower for word in ['date', 'deadline', 'schedule']):
                response += "Important dates and deadlines are mentioned in the document. "
            
            if any(word in context_lower for word in ['application', 'apply', 'form']):
                response += "Application process details are outlined in the document. "
            
            response += "Please refer to the relevant information above for specific details."
            
            # Add sources if available
            if sources:
                response += f"\n\n📋 **Source:** {', '.join(sources)}"
            
            return response
            
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"

def initialize_rag_system_from_text(document_text: str):
    """Initialize RAG system from document text"""
    try:
        if not document_text.strip():
            return None, None, 0
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Process text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(document_text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i, 
                    "source": st.session_state.pdf_filename,
                    "chunk_length": len(chunk)
                }
            )
            documents.append(doc)
        
        # Create vector store
        # Clean up existing vectorstore directory
        db_path = "./chroma_db"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_path
        )
        
        # Create RAG system
        rag_system = SimpleRAGSystem(vectorstore, embeddings)
        
        return rag_system, vectorstore, len(documents)
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None, None, 0

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 University of Punjab Admission Assistant</h1>', unsafe_allow_html=True)
    
    # PDF Upload Section
    if not st.session_state.pdf_processed:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload University Guidelines PDF")
        st.markdown("Please upload the official university admission guidelines PDF to get started.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type=['pdf'],
            help="Upload the official university admission guidelines PDF document"
        )
        
        if uploaded_file is not None:
            st.session_state.pdf_filename = uploaded_file.name
            
            with st.spinner("📖 Processing PDF document..."):
                # Extract text from PDF
                document_text, success = PDFProcessor.process_pdf_file(uploaded_file)
                
                if success and document_text:
                    st.session_state.document_text = document_text
                    
                    # Show document info
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"📊 Document contains {len(document_text)} characters and {len(document_text.split())} words")
                    
                    # Initialize RAG system
                    with st.spinner("🚀 Building knowledge base..."):
                        rag_system, vectorstore, doc_count = initialize_rag_system_from_text(document_text)
                        
                        if rag_system:
                            st.session_state.rag_system = rag_system
                            st.session_state.vectorstore = vectorstore
                            st.session_state.system_initialized = True
                            st.session_state.pdf_processed = True
                            st.success(f"🎉 System ready! Created {doc_count} searchable chunks from your PDF.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize RAG system")
                else:
                    st.error("❌ Failed to extract text from PDF. Please check if the file is valid.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 📋 Document Information")
        
        if st.session_state.pdf_filename:
            st.success(f"📄 Loaded: {st.session_state.pdf_filename}")
            
            # Reset button
            if st.button("🔄 Upload New PDF", type="secondary"):
                # Reset all session state
                for key in ['pdf_processed', 'system_initialized', 'document_text', 'pdf_filename', 'messages', 'rag_system', 'vectorstore']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clean up vectorstore
                db_path = "./chroma_db"
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
                
                st.rerun()
        
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Ask specific questions about admission requirements
        - Inquire about deadlines and important dates  
        - Ask about fee structures and payment details
        - Request information about eligibility criteria
        - Ask about required documents
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 🔧 System Status")
        if st.session_state.system_initialized:
            st.success("✅ RAG System Ready")
            st.info(f"📊 {st.session_state.vectorstore._collection.count() if st.session_state.vectorstore else 0} chunks indexed")
        else:
            st.warning("⏳ System not ready")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat messages
        st.markdown("### 💬 Ask Questions About Your Document")
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>🙋 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        if st.session_state.system_initialized:
            if prompt := st.chat_input("Ask anything about the uploaded document..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response
                with st.spinner("🤔 Searching document..."):
                    response = st.session_state.rag_system.query(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        else:
            st.info("💡 Please upload and process a PDF document first to start chatting!")
    
    with col2:
        # Statistics and info
        st.markdown("### 📊 Chat Statistics")
        
        if st.session_state.system_initialized:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💬 Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📄 Document Chunks", st.session_state.vectorstore._collection.count() if st.session_state.vectorstore else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Sample questions for PDF content
        if st.session_state.system_initialized:
            st.markdown("### 💡 Try These Questions")
            sample_questions = [
                "What are the eligibility requirements?",
                "What is the application process?",
                "What are the important deadlines?",
                "What documents are required?",
                "What are the fees and costs?",
                "How is admission merit calculated?",
                "What programs are offered?",
                "How can I contact the admission office?"
            ]
            
            for i, question in enumerate(sample_questions[:5]):  # Show first 5
                if st.button(f"❓ {question}", key=f"sample_{i}", help="Click to ask this question"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🚀 Built with Streamlit | 🤖 Powered by LangChain & ChromaDB | 📄 PDF-Based RAG System
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()