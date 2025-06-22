#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
from google import genai
from google.genai import types
from datetime import datetime
import time
import subprocess

class VideoSummarizer:
    def __init__(self):
        print("Initializing VideoSummarizer using Gemini API")
        self.client=genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = "gemini-2.5-pro-preview-06-05"
        self.chunk_duration = 30

    def ingest_video(self, video_path):
        print(f"Ingesting video: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            
            video = cv2.VideoCapture(video_path)
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            fps = video.get(cv2.CAP_PROP_FPS)
            size = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_info = {
                'path': video_path,
                'duration': duration,
                'frame_count': frame_count,
                'fps': fps,
                'size': size,
                'filename': os.path.basename(video_path)
            }
            print(f"Duration: {duration:.2f} seconds")
            
            return video, video_info
            
        except Exception as e:
            raise Exception(f"Error loading video: {str(e)}")
    
    def segment_video(self, video_info):
        print(f"Segmenting video into {self.chunk_duration}-second chunks...")
        chunks = []
        duration = video_info['duration']
        chunk_count = int(np.ceil(duration / self.chunk_duration))
        
        for i in range(chunk_count):
            start_time = i * self.chunk_duration
            end_time = min((i + 1) * self.chunk_duration, duration)
            chunk_info = {
                'chunk_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'timestamp': f"{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}"
            }
            
            chunks.append(chunk_info)
            print(f"Chunk {i+1}: {chunk_info['timestamp']}")
        
        print(f"Created {len(chunks)} chunks")
        return chunks

    def extract_video_chunk(self, video_path, chunk_info):
        try:
            os.makedirs("temp", exist_ok=True)
            timestamp = int(time.time()* 1000)
            temp_video_path = f"temp/chunk_{chunk_info['chunk_number']}_{timestamp}.mp4"
            command = [
                'ffmpeg', '-y', 
                '-i', video_path,
                '-ss', str(chunk_info['start_time']),
                '-to', str(chunk_info['end_time']),
                '-c', 'copy', 
                '-y', 
                temp_video_path
            ]
            result = subprocess.run(command, check=True,text=True)
            if result.returncode == 0 and os.path.exists(temp_video_path):
                with open(temp_video_path, 'rb') as f:
                    video_data = f.read()
                return video_data, temp_video_path
            else:
                print(f"Error extracting video chunk {chunk_info['chunk_number']}")
                return None, None

        except Exception as e:
            print(f"Error extracting video chunk {chunk_info['chunk_number']}")
            return None, None

    
    def summarize_chunk(self, video, chunk_info, video_path):
        max_retries = 4
        temp_video_path = None
        e = None 
        for attempt in range(max_retries):
            try:
                video_data, temp_video_path = self.extract_video_chunk(video_path, chunk_info)
                if video_data is None:
                    return "Error extracting video chunk", None
                
                prompt = f"""            
                You are a video summarization expert. Your task is to analyze the provided video frames and audio. Provide a comprehensive summary in English

                Please provide:
                1. Visual description: Describe what is happening in the video (in paragraph). Include details about people, objects, actions, scenes, on-screen text, and any notable visual elements.
                2. Audio content. Provide a detailed explanation of the audio. Do not provide a full transcript. If no audio is present, simply state: “No audio detected.”
                3. Key events: Highlight the main activities or important moments in the video, in chronological order if applicable.
                """
                
                contenttoGemini = []
                contenttoGemini.append(types.Part.from_bytes(
                    data=video_data,
                    mime_type='video/mp4',
                ))
                contenttoGemini.append(types.Part.from_text(text=prompt))

                generation_config = {
                    "max_output_tokens": 4096,
                    "temperature": 0.3
                }

                generatResponse = self.client.models.generate_content(
                    model = self.model,
                    contents= contenttoGemini,
                    config= types.GenerateContentConfig(**generation_config))
                
                summary = generatResponse.text
                print(summary)
                if summary is None or len(summary) == 0:
                    raise ValueError("Received empty summary from Gemini")
                else:
                    summary = summary.strip()
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                else:
                    print(f"Warning: Temporary video file {temp_video_path} not found or already deleted")
                return summary, None
            except Exception as e:
                error_str = str(e)
                print(f"Attempt {attempt + 1} failed for chunk {chunk_info['chunk_number']}: {error_str}")
        
        # Clean up on failure
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        # return f"Error processing chunk {chunk_info['chunk_number']} after {max_retries} attempts: {str(e)}", None
                        
    def create_video_summary_json(self, video_info, chunk_summaries, window_size=180, overlap=60):
        """
        Create video summary JSON with sliding window text chunks to prevent truncation
        when embedding with SentenceTransformer models.
        
        Args:
            video_info: Dictionary containing video metadata
            chunk_summaries: List of tuples (chunk_info, summary)
            window_size: Number of words in each text window
            overlap: Number of words that overlap between windows
        """
        video_summary = {
            'video_name': video_info['filename'],
            'video_path': video_info['path'],
            'total_duration': video_info['duration'],
            'fps': video_info['fps'],
            'size': video_info['size'],
            'processing_date': datetime.now().isoformat(),
            'total_chunks': len(chunk_summaries),
            'chunk_duration': self.chunk_duration,
            'chunks': []
        }
        
        for chunk_info, summary in chunk_summaries:
            # Ensure summary is a string
            if isinstance(summary, list):
                summary = summary[0] if summary else "Error: Empty summary"
            elif not isinstance(summary, str):
                summary = str(summary)
                
            # Create the base chunk data with metadata
            base_chunk_data = {
                'chunk_number': chunk_info['chunk_number'],
                'timestamp': chunk_info['timestamp'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'duration': chunk_info['duration'],
                'summary': summary,  # Keep original for reference
                'summary_length': len(summary),
                'text_windows': []   # New field to store sliding windows
            }
            
            # Apply sliding window to the summary text
            words = summary.split()
            window_count = max(1, (len(words) - overlap) // (window_size - overlap) + 1)
            
            for i in range(window_count):
                start_idx = i * (window_size - overlap)
                end_idx = min(start_idx + window_size, len(words))
                
                # Skip tiny windows at the end
                if end_idx - start_idx < 30 and i > 0:
                    # Extend the previous window instead
                    base_chunk_data['text_windows'][-1]['end_word_idx'] = len(words)
                    base_chunk_data['text_windows'][-1]['text'] = ' '.join(words[
                        base_chunk_data['text_windows'][-1]['start_word_idx']:len(words)
                    ])
                    break
                
                window_text = ' '.join(words[start_idx:end_idx])
                
                # Add window to the chunk
                base_chunk_data['text_windows'].append({
                    'window_number': i + 1,
                    'start_word_idx': start_idx,
                    'end_word_idx': end_idx,
                    'text': window_text,
                    'text_length': len(window_text)
                })
            
            video_summary['chunks'].append(base_chunk_data)
        
        return video_summary
    

    def cleanup_temp_files(self):
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    if file.startswith("chunk_"):
                        os.remove(os.path.join(temp_dir, file))
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"Could not clean up temp directory: {e}")


    def save_summary_json(self, video_summary, output_path=None):
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        
        if output_path is None:
            video_name = os.path.splitext(video_summary['video_name'])[0]
            output_path = os.path.join(output_folder, f"{video_name}_summary.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(video_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {output_path}")
        return output_path
    

    def process_video(self, video_path, output_path=None):
        print("GEMINI CHUNK VIDEO SUMMARIZATION STARTING")
        
        
        try:
            video, video_info = self.ingest_video(video_path)

            chunks = self.segment_video(video_info)

            chunk_summaries = []
            for i, chunk_info in enumerate(chunks):
                try:
                    summary = self.summarize_chunk(video, chunk_info, video_path)

                    if isinstance(summary, tuple):
                        summary, _ = summary 
                        chunk_summaries.append((chunk_info, summary))
                    else:
                        chunk_summaries.append((chunk_info, summary))
                    
                    time.sleep(1)
                
                except Exception as e:
                    if "429" in str(e):
                        print(f"limit hit on chunk {chunk_info['chunk_number']}. Waiting 10 seconds...")
                        time.sleep(10)
                        try:
                            summary = self.summarize_chunk(video, chunk_info, video_path)
                            if isinstance(summary, tuple):
                                summary, _ = summary
                                chunk_summaries.append((chunk_info, summary))
                            else:
                                chunk_summaries.append((chunk_info, summary))
                        except Exception as retry_error:
                            error_summary = f"Error processing chunk: {str(retry_error)}"
                            chunk_summaries.append((chunk_info, error_summary))
                    else:
                        error_summary = f"Error processing chunk: {str(e)}"
                        chunk_summaries.append((chunk_info, error_summary))

            video_summary = self.create_video_summary_json(video_info, chunk_summaries)
            output_file = self.save_summary_json(video_summary, output_path)
            video.release()
            self.cleanup_temp_files()

            print("VIDEO SUMMARIZATION COMPLETED!")
            print(f"Summary saved to: {output_file}")
            print("=" * 50)
            
            return video_summary, output_file
            
        except Exception as e:
            self.cleanup_temp_files()
            print(f"Error processing video: {str(e)}")
            raise
    
