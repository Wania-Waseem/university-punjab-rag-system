# =====================================================
# STREAMLIT WEB INTERFACE FOR RAG SYSTEM WITH PRE-LOADED PDF
# Save this as: streamlit_app.py
# =====================================================

import streamlit as st
import os
import PyPDF2
from io import BytesIO
import numpy as np
from typing import List, Dict
import time
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

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
    .info-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .document-info {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
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
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = 0

# CONFIGURATION: Your PDF file path
PDF_FILE_PATH = "university_guidelines.pdf"  # ← PUT YOUR PDF FILE NAME HERE
DOCUMENT_TITLE = "University of Punjab Admission Guidelines 2024"

class PDFProcessor:
    """Handle PDF file processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except FileNotFoundError:
            st.error(f"❌ PDF file '{pdf_path}' not found. Please make sure the file exists in the same directory as this script.")
            return ""
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> dict:
        """Get PDF information"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return {
                    "pages": len(pdf_reader.pages),
                    "title": pdf_reader.metadata.get('/Title', 'N/A') if pdf_reader.metadata else 'N/A',
                    "author": pdf_reader.metadata.get('/Author', 'N/A') if pdf_reader.metadata else 'N/A',
                    "exists": True
                }
        except:
            return {"exists": False}

class SimpleRAGSystem:
    def __init__(self, vectorstore, embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
    
    def query(self, question: str) -> str:
        """Enhanced RAG query function"""
        try:
            # Search for similar documents
            results = self.vectorstore.similarity_search_with_score(question, k=5)
            
            if not results:
                return "❌ I couldn't find relevant information in the university guidelines. Please try rephrasing your question or ask about topics covered in the admission document."
            
            # Extract relevant documents and scores
            relevant_docs = []
            for doc, score in results:
                # Lower scores mean higher similarity in FAISS
                if score < 1.2:  # Threshold for relevance
                    relevant_docs.append((doc, score))
            
            if not relevant_docs:
                return "❌ I couldn't find sufficiently relevant information for your question. Please try asking about admission requirements, deadlines, fees, or other topics covered in the university guidelines."
            
            # Combine context from retrieved documents
            context_parts = []
            
            for doc, score in relevant_docs:
                context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts)
            
            # Create comprehensive response
            response = f"📄 **Based on the {DOCUMENT_TITLE}:**\n\n"
            
            # Add most relevant context
            if context:
                # Limit context length for readability
                max_context_length = 1200
                truncated_context = context[:max_context_length]
                if len(context) > max_context_length:
                    truncated_context += "...\n\n[Additional relevant information found in guidelines]"
                
                response += f"**📋 Relevant Information:**\n{truncated_context}\n\n"
            
            # Add contextual answer
            response += "💡 **Answer Summary:**\n"
            
            # Enhanced keyword-based responses
            question_lower = question.lower()
            context_lower = context.lower()
            
            # Look for specific information in the context and provide targeted responses
            if any(word in question_lower for word in ['eligibility', 'criteria', 'requirement', 'qualify']):
                if any(word in context_lower for word in ['eligibility', 'criteria', 'requirement']):
                    response += "The eligibility criteria and requirements are detailed above in the relevant information section."
                else:
                    response += "Please check the document for specific eligibility criteria for your program of interest."
                    
            elif any(word in question_lower for word in ['fee', 'cost', 'payment', 'tuition', 'money']):
                if any(word in context_lower for word in ['fee', 'cost', 'payment', 'tuition']):
                    response += "The fee structure and payment information is outlined above."
                else:
                    response += "Please refer to the fee structure section in the complete guidelines document."
                    
            elif any(word in question_lower for word in ['date', 'deadline', 'schedule', 'when', 'timeline']):
                if any(word in context_lower for word in ['date', 'deadline', 'schedule']):
                    response += "Important dates and deadlines are mentioned in the information above."
                else:
                    response += "Please check the academic calendar and important dates section for specific timelines."
                    
            elif any(word in question_lower for word in ['application', 'apply', 'form', 'how to apply']):
                if any(word in context_lower for word in ['application', 'apply', 'form']):
                    response += "The application process is detailed in the relevant information section above."
                else:
                    response += "Please refer to the application procedures section for step-by-step guidance."
                    
            elif any(word in question_lower for word in ['document', 'paper', 'certificate', 'transcript']):
                if any(word in context_lower for word in ['document', 'certificate', 'transcript']):
                    response += "Required documents are listed in the information provided above."
                else:
                    response += "Please check the required documents section for a complete list."
                    
            elif any(word in question_lower for word in ['program', 'course', 'degree', 'major', 'subject']):
                if any(word in context_lower for word in ['program', 'course', 'degree']):
                    response += "Program information is available in the relevant section above."
                else:
                    response += "Please refer to the programs and courses section for detailed information."
                    
            elif any(word in question_lower for word in ['contact', 'phone', 'email', 'address', 'office']):
                if any(word in context_lower for word in ['contact', 'phone', 'email', 'address']):
                    response += "Contact information is provided in the details above."
                else:
                    response += "Please check the contact information section for office details."
            else:
                response += "Based on the information found in the university guidelines, please review the relevant details above."
            
            response += f"\n\n📚 **Source:** {DOCUMENT_TITLE}"
            
            return response
            
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"

@st.cache_resource
def load_embeddings():
    """Load embeddings model with caching"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system from pre-loaded PDF with caching"""
    try:
        # Check if PDF file exists
        if not os.path.exists(PDF_FILE_PATH):
            st.error(f"❌ PDF file '{PDF_FILE_PATH}' not found!")
            st.info("Please make sure to place your PDF file in the same directory as this script.")
            return None, None, 0
        
        # Extract text from PDF
        document_text = PDFProcessor.extract_text_from_pdf(PDF_FILE_PATH)
        
        if not document_text.strip():
            st.error("❌ No text could be extracted from the PDF file.")
            return None, None, 0
        
        # Load embeddings
        embeddings = load_embeddings()
        if embeddings is None:
            return None, None, 0
        
        # Process text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
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
                    "source": DOCUMENT_TITLE,
                    "chunk_length": len(chunk)
                }
            )
            documents.append(doc)
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
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
    
    # Document Information Section
    pdf_info = PDFProcessor.get_pdf_info(PDF_FILE_PATH)
    
    if not pdf_info["exists"]:
        st.markdown('<div class="document-info">', unsafe_allow_html=True)
        st.warning(f"⚠️ **Setup Required:** Please place your PDF file named `{PDF_FILE_PATH}` in the same directory as this application.")
        st.markdown("**Instructions:**")
        st.markdown("1. Save your university admission guidelines PDF file")
        st.markdown(f"2. Rename it to `{PDF_FILE_PATH}`") 
        st.markdown("3. Place it in the same folder as this streamlit app")
        st.markdown("4. Refresh the page")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Initialize RAG system
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Loading University Guidelines..."):
            rag_system, vectorstore, doc_count = initialize_rag_system()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.vectorstore = vectorstore
                st.session_state.document_chunks = doc_count
                st.session_state.system_initialized = True
                st.success(f"✅ System ready! Loaded {doc_count} sections from the university guidelines.")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to initialize the system. Please check the PDF file.")
                return
    
    # Document Status Section
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Document", DOCUMENT_TITLE)
    with col2:
        st.metric("📊 Pages", pdf_info.get("pages", "N/A"))
    with col3:
        st.metric("🧩 Sections", st.session_state.document_chunks)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 📋 About This Assistant")
        st.success("✅ University guidelines loaded and ready!")
        st.info(f"Ask any questions about admission requirements, deadlines, fees, and procedures.")
        
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Ask specific questions about admission requirements
        - Inquire about deadlines and important dates  
        - Ask about fee structures and payment details
        - Request information about eligibility criteria
        - Ask about required documents
        - Inquire about different programs offered
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 🔧 System Status")
        if st.session_state.system_initialized:
            st.success("✅ RAG System Active")
            st.info(f"📊 {st.session_state.document_chunks} sections indexed")
        else:
            st.warning("⏳ Loading system...")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat messages
        st.markdown("### 💬 Ask Questions About University Admissions")
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>🙋 You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Welcome message for first-time users
        if len(st.session_state.messages) == 0:
            st.markdown('<div class="chat-message bot-message"><strong>🤖 Assistant:</strong> Welcome! I can help you with questions about university admissions. Ask me anything about eligibility requirements, deadlines, fees, application procedures, or any other admission-related queries.</div>', unsafe_allow_html=True)
        
        # Chat input
        if st.session_state.system_initialized:
            if prompt := st.chat_input("Ask anything about university admissions..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Generate response
                with st.spinner("🤔 Searching guidelines..."):
                    response = st.session_state.rag_system.query(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        else:
            st.info("💡 System is loading, please wait...")
    
    with col2:
        # Statistics and info
        st.markdown("### 📊 Chat Statistics")
        
        if st.session_state.system_initialized:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💬 Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📄 Sections Available", st.session_state.document_chunks)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Sample questions
        if st.session_state.system_initialized:
            st.markdown("### 💡 Sample Questions")
            sample_questions = [
                "What are the eligibility requirements?",
                "What is the application deadline?",
                "How much are the admission fees?",
                "What documents do I need to submit?",
                "How is merit calculated?",
                "What programs are available?",
                "How can I contact admissions?",
                "What are the scholarship options?"
            ]
            
            for i, question in enumerate(sample_questions[:6]):
                if st.button(f"❓ {question}", key=f"sample_{i}", help="Click to ask this question"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🚀 Built with Streamlit | 🤖 Powered by LangChain & FAISS | 📄 University of Punjab Admission Assistant
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
