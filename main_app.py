# MUST be the very first lines - no other imports above this
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Run startup script to fix SQLite
from startup import setup_environment
setup_environment()

# Clear any modules that might have imported old sqlite3
modules_to_clear = [name for name in sys.modules.keys() 
                   if 'chroma' in name.lower() or 'sqlite' in name.lower()]
for module in modules_to_clear:
    if module not in ['sqlite3', 'sqlite3.dbapi2']:  # Keep our replacements
        sys.modules.pop(module, None)

# NOW import your regular modules
import streamlit as st
from chat_app.client_manager import ClientManager
# ... rest of your imports

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
import semantic_search_app

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
        """Initialize the application's base settings and session state"""
        st.set_page_config(page_title="Multipurpose AI App", layout="wide")
        self.config.setup_directories()

        # Initialize session state for app mode and other variables
        if "app_mode" not in st.session_state:
            st.session_state.app_mode = "Chat with AI"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "last_audio_hash" not in st.session_state:
            st.session_state.last_audio_hash = None
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = self.config.default_system_prompt

    def _initialize_chat_components(self):
        """Initialize components required specifically for the Chat mode"""
        if not self.client_manager.sarvam_client:
            if not self.client_manager.initialize_sarvam_client():
                st.stop()
        if not self.client_manager.llm:
            self.client_manager.initialize_llm(self.config.llm_model, self.config.llm_temperature)

        self.query_transformer = QueryTransformer(self.client_manager.llm)
        self.video_handler = VideoUploadHandler(self.config)
        self.audio_handler = AudioHandler(self.client_manager.sarvam_client)
        self.message_handler = MessageHandler(
            self.client_manager.llm,
            self.query_transformer,
            self.config.default_system_prompt
        )

    def run(self):
        """Run the main application by rendering the sidebar and the selected app mode"""
        self.initialize()
        self._render_sidebar()

        if st.session_state.app_mode == "Chat with AI":
            self._initialize_chat_components()
            self._run_chat_interface()
        elif st.session_state.app_mode == "Semantic Search Comparison":
            # Call the render function from the other app
            semantic_search_app.render_app()

    def _render_sidebar(self):
        """Render the sidebar for mode selection and mode-specific controls"""
        st.sidebar.title("App Mode")
        st.session_state.app_mode = st.sidebar.selectbox(
            "Choose an application",
            ["Chat with AI", "Semantic Search Comparison"],
            key="app_mode_selector"
        )
        st.sidebar.markdown("---")

        if st.session_state.app_mode == "Chat with AI":
            self._render_sidebar_info()

    def _run_chat_interface(self):
        """Run the main chat interface"""
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
