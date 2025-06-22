import os
import streamlit as st
from llm_clients.sarvam_client import SarvamClient
from langchain_google_genai import ChatGoogleGenerativeAI

class ClientManager:
    """Handles initialization of external clients and models"""
    
    def __init__(self):
        self.sarvam_client = None
        self.llm = None
        
    def initialize_sarvam_client(self) -> bool:
        """Initialize SarvamAI client"""
        try:
            self.sarvam_client = SarvamClient()
            st.sidebar.success("SarvamAI Client Initialized.")
            return True
        except ValueError as e:
            st.sidebar.error(f"Failed to initialize SarvamAI Client: {e}")
            return False
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during SarvamAI Client initialization: {e}")
            return False
    
    def initialize_llm(self, model: str, temperature: float) -> bool:
        """Initialize Langchain LLM with Google model currently. This can be extended to other models."""
        try:
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            st.sidebar.info("Langchain LLM initialized successfully.")
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to initialize LLM: {e}")
            return False