import streamlit as st
from config import ChatAppConfig
from chat_app.client_manager import ClientManager
from chat_app.video_handler import VideoUploadHandler
from chat_app.audio_handler import AudioHandler
from chat_app.message_handler import MessageHandler
from chat_app.query_transformation import QueryTransformer
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

class ChatApp:
    """Main chat application class"""
    
    def __init__(self):
        self.config = ChatAppConfig()
        self.client_manager = ClientManager()
        self.query_transformer = None
        self.video_handler = None
        self.audio_handler = None
        self.message_handler = None
        
    def initialize(self):
        """Initialize the application"""
        st.set_page_config(page_title="Chat Application", layout="wide")
        
        self.config.setup_directories()
        
        # Initialize clients
        if not self.client_manager.initialize_sarvam_client():
            st.stop()
        
        self.client_manager.initialize_llm(self.config.llm_model, self.config.llm_temperature)
        
        # Initialize components
        self.query_transformer = QueryTransformer(self.client_manager.llm) # This is the module for query transformation
        self.video_handler = VideoUploadHandler(self.config)
        self.audio_handler = AudioHandler(self.client_manager.sarvam_client)
        # Pass system prompt to message handler
        self.message_handler = MessageHandler(
            self.client_manager.llm, 
            self.query_transformer, 
            self.config.default_system_prompt
        )
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "last_audio_hash" not in st.session_state:
            st.session_state.last_audio_hash = None
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = self.config.default_system_prompt
    
    def run(self):
        """Run the main application"""
        self.initialize()
        
        # Sidebar
        st.sidebar.title("Options")
        # self.video_handler.handle_upload() # WE can enable video upload later, Right now video is processed from terminal
        self._render_sidebar_info()
        
        # Main interface
        st.title("🗣️🎙️ Chat with AI")
        st.markdown("---")
        
        # Display chat
        self.message_handler.display_chat_history()
        
        # Input handling
        self._handle_inputs()
        
        # Auto-scroll to bottom
        self._auto_scroll()
    
    def _handle_inputs(self):
        """Handle user inputs (text and audio)"""
        # Check if we're already processing to prevent multiple simultaneous processing
        if "processing" not in st.session_state:
            st.session_state.processing = False
        
        if st.session_state.processing:
            st.info("Processing your previous input...")
            return
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_text_input = st.chat_input("Type a message or record audio...")
        
        with col2:
            audio_file_path, audio_bytes = self.audio_handler.record_audio()
        
        # Process text input
        if user_text_input:
            st.session_state.processing = True
            self.message_handler.process_text_input(user_text_input)
            st.session_state.processing = False
            st.rerun()
        
        # Process audio input
        if audio_file_path:
            st.session_state.processing = True
            st.info("Audio recorded! Processing...")
            transcribed_text = self.audio_handler.transcribe_audio(audio_file_path)
            self.message_handler.process_audio_input(transcribed_text, audio_bytes)
            st.session_state.processing = False
            st.rerun()
    
    def _render_sidebar_info(self):
        """Render sidebar information and controls"""
        st.sidebar.header("Controls & Info")
        
        st.sidebar.markdown("""
        This chat application allows you to interact via text or voice to search your videos.
        - Type a message in the input box.
        - Click the microphone icon to record audio.
        - Audio is transcribed using **SarvamAI**.
        - Customize the system prompt above to change AI behavior.
        - Use responsibly and have fun!
        """)
        
        if st.sidebar.button("Clear Chat"):
            self.message_handler.clear_chat_history()
            st.rerun()    

    def _auto_scroll(self):
        """Auto-scroll chat to bottom"""
        js_command = "window.scrollTo(0, document.body.scrollHeight);"
        st.markdown(f"<script>{js_command}</script>", unsafe_allow_html=True)


def main():
    """Main entry point"""
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
