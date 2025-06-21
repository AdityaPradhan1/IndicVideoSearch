# sarvam_client.py
import os
from sarvamai import SarvamAI
# We would need a function to save the audio data, e.g.:
from pydub import AudioSegment
import io
import requests

class SarvamClient:
    def __init__(self):
        """
        Initializes the SarvamAI client.
        Requires the SARVAMAI_API_KEY environment variable to be set.
        """
        self.client = SarvamAI(api_subscription_key=os.getenv("SARVAMAI_API_KEY"))
        # Sarvam API Endpoint
        # API endpoint for speech-to-text translation
        self.api_url = "https://api.sarvam.ai/speech-to-text-translate"
        self.api_stt = "https://api.sarvam.ai/speech-to-text"
        # Headers containing the API subscription key
        self.headers = {
            "api-subscription-key": os.getenv("SARVAMAI_API_KEY")
        }
        # Data payload for the translation request
        self.data = {
            "model": "saaras:v2",
            "with_diarization": False  # Set to True for speaker diarization
        }

        self.data_stt = {
            "model": "saarika:v2.5",
            "with_diarization": False  # Set to True for speaker diarization
        }

    # Split audio data into chunks and save to a file
    def split_audio(self, audio_path, chunk_duration_ms):
        """
        Splits an audio file into smaller chunks of specified duration.
        Args:
            audio_path (str): Path to the audio file to be split.
            chunk_duration_ms (int): Duration of each chunk in milliseconds.
        Returns:
            list: A list of AudioSegment objects representing the audio chunks.
        """
        audio = AudioSegment.from_file(audio_path)  # Load the audio file
        chunks = []
        if len(audio) > chunk_duration_ms:
            # Split the audio into chunks of the specified duration
            for i in range(0, len(audio), chunk_duration_ms):
                chunks.append(audio[i:i + chunk_duration_ms])
        else:
            # If the audio is shorter than the chunk duration, use the entire audio
            chunks.append(audio)
        return chunks

    # Translating the audio chunks to text
    def translate_audio(self, audio_file_path, chunk_duration_ms=5*60*1000):
        """
        Translates audio into text with optional diarization and timestamps.
        Args:
            audio_file_path (str): Path to the audio file.
            chunk_duration_ms (int): Duration of each audio chunk in milliseconds.
        Returns:
            dict: Collated response containing the transcript and word-level timestamps.
        """
        # Split the audio into chunks
        chunks = self.split_audio(audio_file_path, chunk_duration_ms)
        responses = []
        # print the number of chunks
        print(f"Number of chunks: {len(chunks)}")
        # Process each chunk
        for idx, chunk in enumerate(chunks):
            # print the size of the chunk and index
            print(f"Processing chunk {idx} of size {len(chunk)} ms")
            # Export the chunk to a BytesIO object (in-memory binary stream)
            chunk_buffer = io.BytesIO()
            chunk.export(chunk_buffer, format="wav")
            chunk_buffer.seek(0)  # Reset the pointer to the start of the stream
            # Prepare the file for the API request
            files = {'file': ('audiofile.wav', chunk_buffer, 'audio/wav')}
            try:
                # Make the POST request to the API
                response = requests.post(self.api_url, headers=self.headers, files=files, data=self.data)
                if response.status_code == 200 or response.status_code == 201:
                    print(f"Chunk {idx} POST Request Successful!")
                    response_data = response.json()
                    transcript = response_data.get("transcript", "")
                    responses.append({"transcript": transcript})
                else:
                    # Handle failed requests
                    print(f"Chunk {idx} POST Request failed with status code: {response.status_code}")
                    print("Response:", response.text)
            except Exception as e:
                # Handle any exceptions during the request
                print(f"Error processing chunk {idx}: {e}")
            finally:
                # Ensure the buffer is closed after processing
                chunk_buffer.close()
        # Collate the transcriptions from all chunks
        collated_transcript = " ".join([resp["transcript"] for resp in responses])
        collated_response = {
            "transcript": collated_transcript,
            "language": response_data.get("language_code")
        }
        return collated_response
    
    def transcribe_audio_stt(self, audio_file_path, chunk_duration_ms=5*60*1000):
        """
        Translates audio into text with optional diarization and timestamps.
        Args:
            audio_file_path (str): Path to the audio file.
            chunk_duration_ms (int): Duration of each audio chunk in milliseconds.
        Returns:
            dict: Collated response containing the transcript and word-level timestamps.
        """
        # Split the audio into chunks
        chunks = self.split_audio(audio_file_path, chunk_duration_ms)
        responses = []
        # print the number of chunks
        print(f"Number of chunks: {len(chunks)}")
        # Process each chunk
        for idx, chunk in enumerate(chunks):
            # print the size of the chunk and index
            print(f"Processing chunk {idx} of size {len(chunk)} ms")
            # Export the chunk to a BytesIO object (in-memory binary stream)
            chunk_buffer = io.BytesIO()
            chunk.export(chunk_buffer, format="wav")
            chunk_buffer.seek(0)  # Reset the pointer to the start of the stream
            # Prepare the file for the API request
            files = {'file': ('audiofile.wav', chunk_buffer, 'audio/wav')}
            try:
                # Make the POST request to the API
                response = requests.post(self.api_stt, headers=self.headers, files=files, data=self.data_stt)
                if response.status_code == 200 or response.status_code == 201:
                    print(f"Chunk {idx} POST Request Successful!")
                    response_data = response.json()
                    transcript = response_data.get("transcript", "")
                    responses.append({"transcript": transcript})
                else:
                    # Handle failed requests
                    print(f"Chunk {idx} POST Request failed with status code: {response.status_code}")
                    print("Response:", response.text)
            except Exception as e:
                # Handle any exceptions during the request
                print(f"Error processing chunk {idx}: {e}")
            finally:
                # Ensure the buffer is closed after processing
                chunk_buffer.close()
        # Collate the transcriptions from all chunks
        collated_transcript = " ".join([resp["transcript"] for resp in responses])
        collated_response = {
            "transcript": collated_transcript,
            "language": response_data.get("language_code")
        }
        return collated_response


    def speech_to_text(self, audio_file_path, translate=True):
        """
        Transcribes speech from an audio file to text using SarvamAI.
        This function expects the path to an audio file.
        The SarvamAI API will attempt to auto-detect the source language and translate to English.

        Args:
            audio_file_path (str): The path to the audio file.

        Returns:
            str: The transcribed text, or an error message if transcription fails.
        """
        try:
            # Open the audio file in binary read mode
            with open(audio_file_path, "rb") as audio_file:
                # Call the SarvamAI speech-to-text API
                if translate:
                    # Translate the audio
                    translation = self.translate_audio(audio_file_path)
                else:
                    # Transcribe the audio
                    translation = self.transcribe_audio_stt(audio_file_path)

                # Display the translation results
                print("Translation Results:")
                print(translation)
            if isinstance(translation, dict) and 'transcript' in translation:
                print(f"Transcription: {translation['transcript']}")
                return translation['transcript']
            else:
                # If the structure is unknown, return the whole response for inspection
                print(f"Unexpected STT response format: {translation}")
                return "Transcription data not found in expected format."

        except FileNotFoundError:
            print(f"Error: Audio file not found at {audio_file_path}")
            return "Error: Audio file not found."
        except Exception as e:
            # Catch any other exceptions during the API call
            print(f"Error during SarvamAI speech-to-text API call: {e}")
            return f"Error during transcription: {e}"

    def chunk_text(self, text, max_length=2000):
        """Splits text into chunks of at most max_length characters while preserving word boundaries."""
        chunks = []
        while len(text) > max_length:
            split_index = text.rfind(" ", 0, max_length)  # Find the last space within limit
            if split_index == -1:
                split_index = max_length  # No space found, force split at max_length
            chunks.append(text[:split_index].strip())  # Trim spaces before adding
            text = text[split_index:].lstrip()  # Remove leading spaces for the next chunk
        if text:
            chunks.append(text.strip())  # Add the last chunk
        return chunks

    def translate_text(self, text, target_language="en", source_language_code="en-IN"):
        """
        Translates text to the target language using SarvamAI.
        This function expects the text to be translated and the target language code.

        Args:
            text (str): The text to translate.
            target_language (str): The target language code (e.g., "en" for English).

        Returns:
            str: The translated text, or an error message if translation fails.
        """
        try:
            # Split the text into chunks of 500 characters
            english_text_chunks = self.chunk_text(text, max_length=500)
            translated_texts = []
            for idx, chunk in enumerate(english_text_chunks):
                response = self.client.text.translate(
                    input=chunk,
                    source_language_code=source_language_code,
                    target_language_code=target_language,
                    speaker_gender="Female",
                    mode="formal",
                    model="sarvam-translate:v1",
                    enable_preprocessing=False,
                )
                translated_text = response.translated_text
                print(f"\n=== Translated Chunk {idx + 1} ===\n{translated_text}\n")
                translated_texts.append(translated_text)
            # Combine all translated chunks
            final_translation = "\n".join(translated_texts)
            return final_translation
        except Exception as e:
            print(f"Error during SarvamAI translation API call: {e}")
            return f"Error during translation: {e}"

    def text_to_speech(self, text, voice="anushka", target_language_code="hi-IN", translate=True):
        """
        Converts text to speech using SarvamAI.
        This function is a placeholder and needs to be expanded based on how
        you want to handle the output (e.g., playing directly or saving to a file).
        https://docs.sarvam.ai/api-reference-docs/cookbook/starter-notebooks/tts-api-tutorial

        Args:
            text (str): The text to convert to speech.
            voice (str): The voice to use for speech synthesis.
            target_language_code (str): The target language code (e.g., "hi-IN" for Hindi).

        Returns:
            The response from the SarvamAI TTS API, or None if an error occurs.
        """
        try:
            # First translate the English text to the target language
            # Translate only if the target language is not English
            if target_language_code == "en-IN" or not translate:
                translated_text = text
            else:
                translated_text = self.translate_text(
                    text=text,
                    target_language=target_language_code,
                )
            # Now convert the translated text to speech
            response = self.client.text_to_speech.convert(
                text=translated_text,
                target_language_code=target_language_code,
                speaker=voice,
                enable_preprocessing=True
            )

            # 'save(response, "output.wav")' can be called to save the audio to a file
            return response
        except Exception as e:
            print(f"Error during SarvamAI text-to-speech API call: {e}")
            return None  # Return None on error
        
