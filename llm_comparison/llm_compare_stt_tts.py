# llm_compare_stt_tts.py

# --- INSTALLATION ---
# 1. Install core libraries (in addition to the ones in requirements.txt):
# pip install -r llm_comparison_requirements.txt
#
# 2. Download NLTK data (needed for tokenization in some metrics):
# Run this in a Python interpreter:
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import sys
sys.path.append('..') # Adds 'parent_folder' to the Python path

import os
import io
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wave
from pydub import AudioSegment
from jiwer import wer, cer
import nltk
from huggingface_hub import InferenceClient

# Import AI Provider Clients
# GEMINI AI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, HttpOptions

# GOOGLE CLOUD text-to-speech as Judge
from google.cloud import speech, texttospeech

# OPENAI
from openai import OpenAI
# Workaround for OpenAI's missing 'wait' and 'stop' functions
from tenacity import retry, stop_after_attempt, wait_random_exponential

# SARVAM AI
from llm_clients.sarvam_client import SarvamClient
from sarvamai.play import save

# AI4BHARAT specific libraries
import torch
import torchaudio
from transformers import AutoTokenizer, SeamlessM4Tv2ForSpeechToText
from transformers import SeamlessM4TTokenizer, SeamlessM4TFeatureExtractor
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf

# --- SETUP ---
# Ensure you have set the following environment variables:
# export SARVAMAI_API_KEY="YOUR_SARVAM_AI_API_KEY"
# export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
# export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google-credentials.json"

# Initialize clients
try:
    sarvam_client = SarvamClient()
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    google_speech_client = speech.SpeechClient()
    # google_tts_client = texttospeech.TextToSpeechClient()
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    hf_client = InferenceClient(token=os.environ.get("HF_TOKEN"))
except Exception as e:
    print(f"Error initializing clients. Please check your API keys and environment variables. Details: {e}")
    exit()

# --- CONFIGURATION ---

# Language code mapping for different providers
LANG_MAP = {
    "malayalam": {
        "sarvam_stt": "ml-IN", "sarvam_tts": "ml-IN",
        "gemini_stt": "ml-IN", "gemini_tts": "ml-IN", # Gemini doesn't support TTS for Malayalam
        "google_stt": "ml-IN", "google_tts": "ml-IN",
        "openai_stt": "ml", "openai_tts": "ml",
        "ai4bharat_stt": "mal", "ai4bharat_tts": "malayalam",
    },
    "hindi": {
        "sarvam_stt": "hi-IN", "sarvam_tts": "hi-IN",
        "gemini_stt": "hi-IN", "gemini_tts": "hi-IN",
        "google_stt": "hi-IN", "google_tts": "hi-IN",
        "openai_stt": "hi", "openai_tts": "hi",
        "ai4bharat_stt": "hin", "ai4bharat_tts": "hindi",
    },
    "english": {
        "sarvam_stt": "en-IN", "sarvam_tts": "en-IN",
        "gemini_stt": "en-IN", "gemini_tts": "en-IN",
        "google_stt": "en-US", "google_tts": "en-IN",
        "openai_stt": "en", "openai_tts": "en",
        "ai4bharat_stt": "eng", "ai4bharat_tts": "english",
    }
}

# Voice ID mapping for TTS providers
VOICE_MAP = {
    "sarvam": {"malayalam": "abhilash", "hindi": "abhilash", "english": "abhilash"},
    "gemini": {"malayalam": "Charon", "hindi": "Charon", "english": "Charon"}, # Gemini TTS does not support Malayalam yet
    "google": {
        "malayalam": "ml-IN-Wavenet-A",
        "hindi": "hi-IN-Wavenet-A", "english": "en-IN-Wavenet-D"
    },
    "openai": {"malayalam": "alloy", "hindi": "alloy", "english": "alloy"},
    "ai4bharat": {"malayalam": "male", "hindi": "male", "english": "male"}
}

# --- DATA PREPARATION ---

def load_audio_data(language, audio_dir="audio_samples"):
    """Loads audio file and its reference transcript."""
    audio_path = os.path.join(audio_dir, f"{language}_sample.wav")
    ref_transcript_path = os.path.join(audio_dir, f"{language}_reference_stt.txt")

    if not os.path.exists(audio_path) or not os.path.exists(ref_transcript_path):
        print(f"Warning: Audio sample or reference transcript not found for {language}.")
        return None, ""

    with open(ref_transcript_path, "r", encoding="utf-8") as f:
        reference_stt = f.read().strip()
    return audio_path, reference_stt

# --- METRIC CALCULATION ---

def normalize_text(text):
    """
    Normalizes text for consistent metric calculation by lowercasing,
    removing punctuation, and standardizing whitespace.
    """
    from jiwer import transforms as tr
    normalize_transform = tr.Compose([
        tr.ToLowerCase(),
        tr.RemovePunctuation(),
        tr.RemoveMultipleSpaces(),
        tr.Strip()
    ])
    return normalize_transform(text)

def calculate_wer_cer(reference, hypothesis):
    """Calculates Word Error Rate (WER) and Character Error Rate (CER)."""
    if not reference or not hypothesis or "Error:" in hypothesis:
        return float('nan'), float('nan')
    normalized_ref = normalize_text(reference)
    normalized_hyp = normalize_text(hypothesis)
    
    _wer = wer(normalized_ref, normalized_hyp)
    _cer = cer(normalized_ref, normalized_hyp)
    return _wer, _cer

def save_audio_bytes(audio_data, filename):
    """Saves audio data (bytes) to a file."""
    with open(filename, "wb") as f:
        f.write(audio_data)
    print(f"Audio saved to {filename}")

def save_gemini_audio(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """Saves PCM audio data from Gemini to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    print(f"Audio saved to {filename}")

# --- STT (SPEECH-TO-TEXT) FUNCTIONS ---

async def sarvam_stt(audio_path, lang_code):
    """Sarvam AI STT"""
    try:

        response = sarvam_client.speech_to_text(
            audio_file_path=audio_path,
            translate = False
        )
        return response
    except Exception as e:
        return f"Sarvam STT Error: {e}"

async def google_stt(audio_path, lang_code):
    """Google Cloud Speech-to-Text"""
    try:
        audio_segment = AudioSegment.from_file(audio_path)
        sample_rate = audio_segment.frame_rate

        # Convert to mono if stereo
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
            # Export mono audio to an in-memory bytes buffer
            mono_audio_io = io.BytesIO()
            audio_segment.export(mono_audio_io, format="wav")
            mono_audio_io.seek(0)
            content = mono_audio_io.read()
        else:
            # If already mono, just read the file content
            with open(audio_path, "rb") as audio_file:
                content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=lang_code,
        )
        response = google_speech_client.recognize(config=config, audio=audio)
        return " ".join([result.alternatives[0].transcript for result in response.results])
    except Exception as e:
        return f"Google STT Error: {e}"

async def openai_stt(audio_path, lang_code):
    """OpenAI Whisper STT with retry logic."""

    # Define an inner async function to apply the retry decorator to
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def _transcribe_with_retry():
        with open(audio_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=lang_code
            )
        return response.text

    try:
        transcript = await _transcribe_with_retry()
        return transcript
    except Exception as e:
        # This exception will be raised if all retries fail
        return f"OpenAI STT Error: {e}"

async def gemini_stt(audio_path, lang_code):
    """Gemini STT"""
    try:
        gemini_file = await asyncio.to_thread(gemini_client.files.upload, file=audio_path)
        prompt = f"Convert this speech to text in {lang_code} Language:"
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, gemini_file]
        )
        # Clean up the uploaded file after transcription
        # genai.delete_file(gemini_file.name)

        print("Gemini STT: ", response.text)
        return response.text
    except Exception as e:
        return f"Gemini STT Error: {e}"

async def ai4bharat_stt(audio_path, lang_code):
    """AI4Bharat STT (IndicASR) via Hugging Face"""
    try:
        # This function is CPU/GPU bound, running it in a thread is a good practice for async
        def _transcribe():
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless")
            processor = SeamlessM4TFeatureExtractor.from_pretrained("ai4bharat/indic-seamless")
            tokenizer = SeamlessM4TTokenizer.from_pretrained("ai4bharat/indic-seamless")

            audio, orig_freq = torchaudio.load(audio_path)
            audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
            audio_inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

            text_out = model.generate(**audio_inputs, tgt_lang=lang_code)[0].numpy().squeeze()
            final_text = tokenizer.decode(text_out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            print("ai4bharat STT: ", final_text)
            return final_text
        
        return await asyncio.to_thread(_transcribe)
    except Exception as e:
        return f"AI4Bharat STT Error: {e}"

# --- TTS (TEXT-TO-SPEECH) FUNCTIONS ---

async def sarvam_tts(text, lang_code, speaker_id):
    """Sarvam AI TTS"""
    try:
        response = sarvam_client.text_to_speech(
            text=text,
            target_language_code=lang_code,
            voice=speaker_id,
            translate=False
        )
        return response
    except Exception as e:
        print(f"Sarvam TTS Error: {e}")
        return None

async def gemini_tts(text, lang_code, voice_name):
    """Gemini TTS"""
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents="Say cheerfully in the language " + lang_code + ":" + text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            )
        )
        return response.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        print(f"Gemini TTS Error: {e}")
        return None


# async def google_tts(text, lang_code, voice_name):
#     """Google Cloud TTS"""
#     try:
#         synthesis_input = texttospeech.SynthesisInput(text=text)
#         voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
#         audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
#         response = google_tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
#         return response.audio_content
#     except Exception as e:
#         print(f"Google TTS Error: {e}")
#         return None

async def openai_tts(text, voice_name):
    """OpenAI TTS"""
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text,
            response_format="wav"
        )
        return response.content
    except Exception as e:
        print(f"OpenAI TTS Error: {e}")
        return None

async def ai4bharat_tts(text, lang_code, speaker_id):
    """AI4Bharat TTS (Indic Parler TTS) via Hugging Face using generic post."""
    try:
        # This function is CPU/GPU bound, running it in a thread is a good practice for async
        def _synthesize():
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
            tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

            description = f'''A {speaker_id} speaker delivers a slightly expressive and animated 
                            speech in {lang_code} with a moderate speed and pitch. 
                            The recording is of very high quality, with the speaker's 
                            voice sounding clear and very close up.'''

            description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

            generation = model.generate(
                input_ids=description_input_ids.input_ids,
                attention_mask=description_input_ids.attention_mask,
                prompt_input_ids=prompt_input_ids.input_ids,
                prompt_attention_mask=prompt_input_ids.attention_mask)
            audio_arr = generation.cpu().numpy().squeeze()

            return audio_arr, model.config.sampling_rate
        
        return await asyncio.to_thread(_synthesize)
    except Exception as e:
        print(f"AI4Bharat TTS Error: {e}")
        return None, None

# --- VISUALIZATION ---

def plot_results(df):
    """Generates and saves detailed bar plots for STT and TTS results."""
    phases = ['STT', 'TTS_Intelligibility']
    metrics = ['WER', 'CER']

    for phase in phases:
        for metric in metrics:
            plot_df = df[(df['Phase'] == phase) & (df['Metric'] == metric)].dropna(subset=['Value'])
            if not plot_df.empty:
                plt.figure(figsize=(12, 7))
                sns.barplot(data=plot_df, x='Language', y='Value', hue='Provider', palette='viridis')
                
                title_phase = "Speech-to-Text (STT)" if phase == 'STT' else "TTS Intelligibility (Judged by Google Cloud STT)"
                title_metric = "Word Error Rate (WER)" if metric == 'WER' else "Character Error Rate (CER)"
                
                plt.title(f'{title_phase} Comparison - {title_metric}', fontsize=16)
                plt.ylabel(f'{title_metric} (Lower is Better)', fontsize=12)
                plt.xlabel('Language', fontsize=12)
                plt.legend(title='Provider')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                filename = f'{phase.lower()}_{metric.lower()}_comparison.png'
                plt.savefig(filename)
                print(f"\nComparison plot saved to {filename}")
                plt.show()


def display_tables(df):
    """Pivots the DataFrame to create and print summary tables for WER and CER."""
    # STT WER Table
    stt_wer_table = df[(df['Phase'] == 'STT') & (df['Metric'] == 'WER')].pivot_table(
        index='Language', columns='Provider', values='Value'
    )
    print("\n--- STT Word Error Rate (WER) Summary Table ---")
    print(stt_wer_table.to_markdown(floatfmt=".4f"))

    # STT CER Table
    stt_cer_table = df[(df['Phase'] == 'STT') & (df['Metric'] == 'CER')].pivot_table(
        index='Language', columns='Provider', values='Value'
    )
    print("\n--- STT Character Error Rate (CER) Summary Table ---")
    print(stt_cer_table.to_markdown(floatfmt=".4f"))

    # TTS Intelligibility WER Table
    tts_wer_table = df[(df['Phase'] == 'TTS_Intelligibility') & (df['Metric'] == 'WER')].pivot_table(
        index='Language', columns='Provider', values='Value'
    )
    print("\n--- TTS Intelligibility WER Summary Table (Judged by ASR) ---")
    print(tts_wer_table.to_markdown(floatfmt=".4f"))

    # TTS Intelligibility CER Table
    tts_cer_table = df[(df['Phase'] == 'TTS_Intelligibility') & (df['Metric'] == 'CER')].pivot_table(
        index='Language', columns='Provider', values='Value'
    )
    print("\n--- TTS Intelligibility CER Summary Table (Judged by ASR) ---")
    print(tts_cer_table.to_markdown(floatfmt=".4f"))


# --- MAIN EVALUATION LOOP ---

async def main_evaluation_loop():
    languages = ["malayalam", "hindi", "english"]
    providers = ["sarvam", "gemini", "ai4bharat"] # 'openai' removed due to API limitations

    results = []

    for lang in languages:
        print(f"\n{'='*20} Processing Language: {lang.upper()} {'='*20}")
        
        # 1. Prepare Data
        audio_path, reference_stt = load_audio_data(lang)
        if not audio_path:
            print(f"Skipping {lang} due to missing data.")
            continue
        
        # 2. STT Phase
        stt_transcripts = {}
        for provider in providers:
            print(f"  -> Performing STT with {provider}...")
            if f"{provider}_stt" not in LANG_MAP[lang]:
                print(f"     Skipping STT for {provider} in {lang} - no STT config.")
                continue

            stt_lang_code = LANG_MAP[lang][f"{provider}_stt"]
            transcript = f"Error: STT for {provider} not processed"
            
            try:
                if provider == "sarvam":
                    transcript = await sarvam_stt(audio_path, stt_lang_code)
                elif provider == "gemini":
                    transcript = await gemini_stt(audio_path, stt_lang_code)
                elif provider == "google":
                    transcript = await google_stt(audio_path, stt_lang_code)
                elif provider == "openai":
                    transcript = await openai_stt(audio_path, stt_lang_code)
                elif provider == "ai4bharat":
                    transcript = await ai4bharat_stt(audio_path, stt_lang_code)
            except Exception as e:
                transcript = f"Error during STT call for {provider}: {e}"

            stt_transcripts[provider] = transcript
            wer_val, cer_val = calculate_wer_cer(reference_stt, transcript)
            
            print(f"     Reference : {reference_stt}")
            print(f"     Transcript: {transcript}")
            print(f"     {provider} STT - WER: {wer_val:.4f}, CER: {cer_val:.4f}")

            results.append({"Phase": "STT", "Language": lang, "Provider": provider, "Metric": "WER", "Value": wer_val})
            results.append({"Phase": "STT", "Language": lang, "Provider": provider, "Metric": "CER", "Value": cer_val})

        # 3. TTS Phase (using the reference transcript as input)
        tts_audio_dir = "tts_outputs"
        os.makedirs(tts_audio_dir, exist_ok=True)
        
        for provider in providers:
            print(f"  -> Performing TTS with {provider}...")
            tts_lang_code = LANG_MAP[lang][f"{provider}_tts"]
            speaker_id = VOICE_MAP[provider].get(lang)

            if not speaker_id:
                print(f"     Skipping TTS for {provider} in {lang} - no voice configured.")
                continue

            audio_content = None
            sampling_rate = None # For AI4Bharat
            
            try:
                if provider == "sarvam":
                    audio_content = await sarvam_tts(reference_stt, tts_lang_code, speaker_id)
                elif provider == "gemini":
                    audio_content = await gemini_tts(reference_stt, tts_lang_code, speaker_id)
                elif provider == "google":
                    audio_content = await google_tts(reference_stt, tts_lang_code, speaker_id)
                elif provider == "openai":
                    audio_content = await openai_tts(reference_stt, speaker_id) # Lang not needed for OpenAI model
                elif provider == "ai4bharat":
                    audio_content, sampling_rate = await ai4bharat_tts(reference_stt, lang, speaker_id) # Uses lang name
            except Exception as e:
                print(f"     Error during TTS call for {provider}: {e}")

            if audio_content is not None:
                tts_output_path = os.path.join(tts_audio_dir, f"{provider}_{lang}_tts_output.wav")
                if provider == "sarvam":
                    save(audio_content, tts_output_path)
                elif provider == "gemini":
                    save_gemini_audio(tts_output_path, audio_content)
                elif provider == "ai4bharat":
                    sf.write(tts_output_path, audio_content, sampling_rate)
                else:
                    save_audio_bytes(audio_content, tts_output_path)

                # Evaluate TTS intelligibility using a single, objective ASR 'judge'.
                # We use Google STT here for all providers to ensure a fair comparison.
                print(f" ======== Evaluating TTS intelligibility with Google STT as the judge =========")
                judge_lang_code = LANG_MAP[lang]["google_stt"]
                tts_stt_transcript = await google_stt(tts_output_path, judge_lang_code)

                wer_tts, cer_tts = calculate_wer_cer(reference_stt, tts_stt_transcript)
                print(f"     Original Text    : {reference_stt}")
                print(f"     TTS->STT Result  : {tts_stt_transcript}")
                print(f"     {provider} TTS Intelligibility - WER: {wer_tts:.4f}, CER: {cer_tts:.4f}")

                results.append({"Phase": "TTS_Intelligibility", "Language": lang, "Provider": provider, "Metric": "WER", "Value": wer_tts})
                results.append({"Phase": "TTS_Intelligibility", "Language": lang, "Provider": provider, "Metric": "CER", "Value": cer_tts})
            else:
                print(f"     {provider} TTS failed to generate audio for {lang}.")
                results.append({"Phase": "TTS_Intelligibility", "Language": lang, "Provider": provider, "Metric": "WER", "Value": float('nan')})
                results.append({"Phase": "TTS_Intelligibility", "Language": lang, "Provider": provider, "Metric": "CER", "Value": float('nan')})

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Allows running asyncio loops in environments that might already have one.
    import nest_asyncio
    nest_asyncio.apply()
    
    # Run the main evaluation
    evaluation_df = asyncio.run(main_evaluation_loop())
    
    print(f"\n\n{'='*25} FINAL RESULTS {'='*25}")
    
    # Display results in clean tables
    display_tables(evaluation_df)
    
    # Generate and save plots
    plot_results(evaluation_df)
