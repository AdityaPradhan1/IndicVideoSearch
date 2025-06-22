#!/usr/bin/env python3
"""
Video RAG System - Video Processing Module
Automatically processes videos in any language and creates searchable summaries
"""

import os
import json
import cv2
import numpy as np
from openai import OpenAI
from datetime import datetime
import base64
import time
import subprocess
from llm_clients.sarvam_client import SarvamClient


class VideoSummarizer:
    def __init__(self):
        print("Initializing VideoSummarizer using OpenAI and Sarvam AI")
        self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.sarvam_client = SarvamClient()
        self.model = "gpt-4o"
        self.api_url = "https://api.sarvam.ai/speech-to-text-translate"
        self.headers = {
            "api-subscription-key": os.getenv("SARVAMAI_API_KEY")
        }
        self.data = {
            "model": "saaras:v2",
            "with_diarization": False
        }
        # Configuration
        self.chunk_duration = 30
        self.max_summary_chars = 1500

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
            
            print(f"Video loaded successfully!")
            print(f"Duration: {duration:.2f} seconds")
            print(f"FPS: {fps}")
            print(f"Size: {size}")
            
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

    def extract_frames_and_audio(self, chunk_info, video_path):
        """Extract key frames and audio from a video chunk"""

        temp_video_path = None
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

        except Exception as e:
            print(f"Error extracting video chunk {chunk_info['chunk_number']}")
        
        if temp_video_path and os.path.exists(temp_video_path):
            cap = cv2.VideoCapture(temp_video_path)
        else:
            raise FileNotFoundError(f"Temporary video file not found for chunk {chunk_info['chunk_number']}")
        print(f"Extracting frames from video: {temp_video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        max_frame = 15
        if total_frames > max_frame:
            interval = total_frames // max_frame 
        else:
            interval = 1
        base64Frames = []
        frame_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % interval == 0:
                frame = cv2.resize(frame, (480, 270))
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))    
            frame_counter += 1
            #save frame to folder
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_folder = f"frames/{video_name}"
        os.makedirs(save_folder, exist_ok=True)
        for i, frame in enumerate(base64Frames):
            frame_data = base64.b64decode(frame)
            with open(os.path.join(save_folder, f"frame_{i:04d} {chunk_info['chunk_number']} .jpg"), "wb") as f:
                f.write(frame_data)
        cap.release()

        # Extract audio using FFmpeg
        audioTranslate = None
        temp_audio_path = None
        
        try:
            os.makedirs("temp", exist_ok=True)
            timestamp = int(time.time() * 1000)
            temp_audio_path = f"temp/chunk_audio_{timestamp}.wav"
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(chunk_info['start_time']),
                '-t', str(chunk_info['duration']),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', '-y',
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_audio_path):
                audioTranslate = self.sarvam_client.translate_audio(temp_audio_path)
                print(audioTranslate)

            else:
                print(f"Warning: Could not extract audio for chunk {chunk_info['chunk_number']}")

        except Exception as e:
            print(f"Warning: Could not extract audio - {str(e)}")
            audioTranslate = None
            temp_audio_path = None
        return base64Frames, audioTranslate, temp_audio_path

        

    
    def summarize_chunk(self, video, chunk_info, video_path):
        """Function 3: Summarize video chunk using Gemini"""
        print(f"Summarizing chunk {chunk_info['chunk_number']}...")
        
        try:
            base64Frames, audioTranslate, temp_audio_path = self.extract_frames_and_audio(chunk_info, video_path)
            print(f"Extracted {len(base64Frames)} frames for chunk {chunk_info['chunk_number']}")
                
            prompt = f"""            
            You are a video summarization expert. Your task is to analyze the provided video frames and audio. Provide a comprehensive summary in English

            Please provide:
            1. Visual description: Describe what is happening in the video (in paragraph). Include details about people, objects, actions, scenes, on-screen text, and any notable visual elements.
            2. Audio content. Provide a detailed explanation of the audio. Do not provide a full transcript. If no audio is present, simply state: “No audio detected.”
            3. Key events: Highlight the main activities or important moments in the video, in chronological order if applicable.
            Audio transcription: "{audioTranslate}"
            """
            
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        prompt,
                        *map(lambda x: {"image": x, "resize": 480}, base64Frames),
                    ],
                },
            ]
            params = {
                "model": self.model,
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1000,
                "temperature": 0.3,
            }

            response = self.client.chat.completions.create(**params)
            summary = response.choices[0].message.content 
            print(summary)

            if summary is None or summary.strip() == "":
                raise ValueError("empty summary")  
            else:
                summary = summary.strip()
            
            print(f"Summary generated ({len(summary)} characters)")
            
            # Clean up temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp audio file {temp_audio_path}: {e}")
            return summary
            
        except Exception as e:
            print(f"Error summarizing chunk {chunk_info['chunk_number']}: {str(e)}")
            return f"Error processing chunk: {str(e)}"
    
    def cleanup_temp_files(self):
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    if file.startswith("chunk_audio_"):
                        os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")
                
    def create_video_summary_json(self, video_info, chunk_summaries):
        print("Creating video summary JSON...")
        
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
            chunk_data = {
                'chunk_number': chunk_info['chunk_number'],
                'timestamp': chunk_info['timestamp'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'duration': chunk_info['duration'],
                'summary': summary,
                'summary_length': len(summary),
            }
            video_summary['chunks'].append(chunk_data)
        
        return video_summary
    
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
        """Main function to process entire video and create JSON summary"""
        
        try:
            #Ingest video
            video, video_info = self.ingest_video(video_path)
            
            #Segment video
            chunks = self.segment_video(video_info)
            
            # Summarize each chunk with rate limiting
            chunk_summaries = []
            for i, chunk_info in enumerate(chunks):
                try:
                    summary = self.summarize_chunk(video, chunk_info, video_path)
                    chunk_summaries.append((chunk_info, summary))
                    
                    time.sleep(3)
                
                except Exception as e:
                    if "429" in str(e):
                        print(f"limit hit on chunk {chunk_info['chunk_number']}. Waiting 10 seconds...")
                        time.sleep(10)
                        try:
                            summary = self.summarize_chunk(video, chunk_info, video_path)
                            chunk_summaries.append((chunk_info, summary))
                        except Exception as retry_error:
                            error_summary = f"Error processing chunk: {str(retry_error)}"
                            chunk_summaries.append((chunk_info, error_summary, None))
                    else:
                        error_summary = f"Error processing chunk: {str(e)}"
                        chunk_summaries.append((chunk_info, error_summary, None))
            
            # JSON summary
            video_summary = self.create_video_summary_json(video_info, chunk_summaries)
            
            # Save to file
            output_file = self.save_summary_json(video_summary, output_path)
            
            # Clean up
            video.release()
            self.cleanup_temp_files()
            
            print("=" * 50)
            print("VIDEO SUMMARIZATION COMPLETED!")
            print(f"Summary saved to: {output_file}")
            print("=" * 50)
            
            return video_summary, output_file
            
        except Exception as e:
            self.cleanup_temp_files()
            print(f"Error processing video: {str(e)}")
            raise