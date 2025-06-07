#!/usr/bin/env python3
"""
Video RAG System - Video Processing Module
Automatically processes videos in any language and creates searchable summaries
"""

import os
import json
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import google.generativeai as genai
from datetime import datetime
import base64
import time

class VideoSummarizer:
    def __init__(self):
        # Configure Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Configuration
        self.chunk_duration = 30  # 30 seconds per chunk
        self.max_summary_chars = 1500
        
    def ingest_video(self, video_path):
        """Function 1: Ingest/upload video file"""
        print(f"Ingesting video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            video = VideoFileClip(video_path)
            video_info = {
                'path': video_path,
                'duration': video.duration,
                'fps': video.fps,
                'size': video.size,
                'filename': os.path.basename(video_path)
            }
            
            print(f"Video loaded successfully!")
            print(f"Duration: {video.duration:.2f} seconds")
            print(f"FPS: {video.fps}")
            print(f"Size: {video.size}")
            
            return video, video_info
            
        except Exception as e:
            raise Exception(f"Error loading video: {str(e)}")
    
    def segment_video(self, video, video_info):
        """Function 2: Segment video into chunks"""
        print(f"Segmenting video into {self.chunk_duration}-second chunks...")
        
        chunks = []
        duration = video_info['duration']
        chunk_count = int(np.ceil(duration / self.chunk_duration))
        
        for i in range(chunk_count):
            start_time = i * self.chunk_duration
            end_time = min((i + 1) * self.chunk_duration, duration)
            
            chunk = video.subclip(start_time, end_time)
            
            chunk_info = {
                'chunk_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'timestamp': f"{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}"
            }
            
            chunks.append((chunk, chunk_info))
            print(f"Chunk {i+1}: {chunk_info['timestamp']}")
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def extract_frames_and_audio(self, chunk):
        """Extract key frames from a video chunk"""
        duration = chunk.duration
        frame_times = [0, duration/2, duration-0.1] if duration > 0.1 else [0]
        
        frames = []
        for t in frame_times:
            if t < duration:
                frame = chunk.get_frame(t)
                frames.append(frame)
        
        return frames, None
    
    def frames_to_base64(self, frames):
        """Convert frames to base64 for API"""
        base64_frames = []
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_b64)
        
        return base64_frames
    
    def summarize_chunk(self, chunk, chunk_info):
        """Function 3: Summarize video chunk using Gemini"""
        print(f"Summarizing chunk {chunk_info['chunk_number']}...")
        
        try:
            frames, _ = self.extract_frames_and_audio(chunk)
            
            prompt = f"""
            Analyze this {self.chunk_duration}-second video segment and provide a detailed summary in English.
            
            Time range: {chunk_info['timestamp']}
            Chunk duration: {chunk_info['duration']:.2f} seconds
            
            Please provide:
            1. Visual description: What is happening in the video? Include objects, people, actions, scenes, text if any
            2. Audio analysis: Describe any speech, music, sound effects, or ambient sounds
            3. Key events: Main activities or important moments in this segment
            4. Context: Overall theme or topic of this segment
            
            Keep the summary detailed but concise (max {self.max_summary_chars} characters).
            Focus on the most important visual and audio elements.
            """
            
            base64_frames = self.frames_to_base64(frames)
            content = [prompt]
            
            for frame_b64 in base64_frames:
                content.append({
                    "mime_type": "image/jpeg",
                    "data": frame_b64
                })
            
            response = self.model.generate_content(content)
            summary = response.text
            
            if len(summary) > self.max_summary_chars:
                summary = summary[:self.max_summary_chars-3] + "..."
            
            return summary
            
        except Exception as e:
            print(f"Error summarizing chunk {chunk_info['chunk_number']}: {str(e)}")
            return f"Error processing chunk: {str(e)}"
    
    def create_video_summary_json(self, video_info, chunk_summaries):
        """Function 4: Create JSON with video summary data"""
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
                'summary_length': len(summary)
            }
            video_summary['chunks'].append(chunk_data)
        
        return video_summary
    
    def save_summary_json(self, video_summary, output_path=None):
        """Save the video summary to JSON file"""
        # Ensure output folder exists
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        
        if output_path is None:
            video_name = os.path.splitext(video_summary['video_name'])[0]
            output_path = os.path.join(output_folder, f"{video_name}_summary.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(video_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {output_path}")
        return output_path
    
    def check_existing_summary(self, video_path):
        """Check if a summary already exists for this video"""
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        summary_filename = f"{video_name}_summary.json"
        
        if os.path.exists(summary_filename):
            try:
                with open(summary_filename, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
                return existing_summary, summary_filename
            except (json.JSONDecodeError, FileNotFoundError):
                return None, None
        
        return None, None
    
    def load_existing_summary(self, summary_path):
        """Load and return an existing summary"""
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            return summary
        except Exception as e:
            print(f"Error loading existing summary: {e}")
            return None

    def resume_failed_processing(self, video_path, existing_summary_path):
        """Resume processing from where it failed due to rate limits"""
        print("🔄 RESUMING FAILED PROCESSING")
        print("=" * 50)
        
        try:
            # Load existing summary
            with open(existing_summary_path, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
            
            # Check which chunks failed
            failed_chunks = []
            for chunk in existing_summary['chunks']:
                if 'Error processing chunk' in chunk.get('summary', ''):
                    failed_chunks.append(chunk['chunk_number'])
            
            if not failed_chunks:
                print("✅ No failed chunks found. Summary appears complete!")
                return existing_summary, existing_summary_path
            
            print(f"🔍 Found {len(failed_chunks)} failed chunks: {failed_chunks}")
            print("⏳ Re-processing failed chunks...")
            
            # Re-process the video
            video, video_info = self.ingest_video(video_path)
            chunks = self.segment_video(video, video_info)
            
            # Only re-process failed chunks
            for chunk, chunk_info in chunks:
                if chunk_info['chunk_number'] in failed_chunks:
                    print(f"🔄 Re-processing chunk {chunk_info['chunk_number']}...")
                    
                    # Add longer delay for rate limit recovery
                    time.sleep(3)
                    
                    try:
                        summary = self.summarize_chunk(chunk, chunk_info)
                        
                        # Update the existing summary
                        for i, existing_chunk in enumerate(existing_summary['chunks']):
                            if existing_chunk['chunk_number'] == chunk_info['chunk_number']:
                                existing_summary['chunks'][i]['summary'] = summary
                                existing_summary['chunks'][i]['summary_length'] = len(summary)
                                break
                        
                        print(f"✅ Successfully re-processed chunk {chunk_info['chunk_number']}")
                        
                    except Exception as e:
                        print(f"❌ Still failing on chunk {chunk_info['chunk_number']}: {str(e)}")
                        if "429" in str(e):
                            print("⏸️  Rate limit hit again. Waiting 60 seconds...")
                            time.sleep(60)
                            try:
                                summary = self.summarize_chunk(chunk, chunk_info)
                                for i, existing_chunk in enumerate(existing_summary['chunks']):
                                    if existing_chunk['chunk_number'] == chunk_info['chunk_number']:
                                        existing_summary['chunks'][i]['summary'] = summary
                                        existing_summary['chunks'][i]['summary_length'] = len(summary)
                                        break
                                print(f"✅ Successfully re-processed chunk {chunk_info['chunk_number']} after retry")
                            except Exception as retry_error:
                                print(f"❌ Final failure on chunk {chunk_info['chunk_number']}: {str(retry_error)}")
            
            # Update metadata
            existing_summary['processing_date'] = datetime.now().isoformat()
            
            # Save updated summary
            output_file = self.save_summary_json(existing_summary, existing_summary_path)
            
            # Clean up
            video.close()
            for chunk, _ in chunks:
                chunk.close()
            
            print("=" * 50)
            print("RESUME PROCESSING COMPLETED!")
            print(f"Updated summary saved to: {output_file}")
            print("=" * 50)
            
            return existing_summary, output_file
            
        except Exception as e:
            print(f"Error resuming processing: {str(e)}")
            raise

    def process_video(self, video_path, output_path=None, force_reprocess=False):
        """Main function to process entire video and create JSON summary"""
        print("=" * 50)
        print("VIDEO SUMMARIZATION STARTING")
        print("=" * 50)
        
        # Check if summary already exists
        if not force_reprocess:
            existing_summary, existing_path = self.check_existing_summary(video_path)
            if existing_summary:
                # Check if there are any failed chunks
                failed_chunks = []
                for chunk in existing_summary.get('chunks', []):
                    if 'Error processing chunk' in chunk.get('summary', ''):
                        failed_chunks.append(chunk['chunk_number'])
                
                if failed_chunks:
                    print("🔍 FOUND EXISTING SUMMARY WITH FAILED CHUNKS!")
                    print(f"📄 Summary file: {existing_path}")
                    print(f"❌ Failed chunks: {failed_chunks}")
                    print(f"📅 Previously processed: {existing_summary.get('processing_date', 'Unknown')}")
                    print()
                    
                    resume_choice = input("💡 Resume processing failed chunks? (y/n): ").lower().strip()
                    if resume_choice == 'y' or resume_choice == 'yes' or resume_choice == '':
                        return self.resume_failed_processing(video_path, existing_path)
                    else:
                        print("🔄 Full re-processing selected...")
                else:
                    print("🔍 FOUND EXISTING SUMMARY!")
                    print(f"📄 Summary file: {existing_path}")
                    print(f"📅 Previously processed: {existing_summary.get('processing_date', 'Unknown')}")
                    print(f"⏱️  Video duration: {existing_summary.get('total_duration', 0):.1f} seconds")
                    print(f"📦 Total chunks: {existing_summary.get('total_chunks', 0)}")
                    print()
                    
                    user_choice = input("💡 Summary already exists. Use existing? (y/n): ").lower().strip()
                    
                    if user_choice == 'y' or user_choice == 'yes' or user_choice == '':
                        print("✅ Using existing summary (no API calls needed)")
                        print("=" * 50)
                        print("VIDEO SUMMARIZATION COMPLETED!")
                        print(f"Summary loaded from: {existing_path}")
                        print("=" * 50)
                        return existing_summary, existing_path
                    
                    print("🔄 Re-processing video as requested...")
        
        try:
            # Step 1: Ingest video
            video, video_info = self.ingest_video(video_path)
            
            # Step 2: Segment video
            chunks = self.segment_video(video, video_info)
            
            # Step 3: Summarize each chunk with rate limiting
            chunk_summaries = []
            for i, (chunk, chunk_info) in enumerate(chunks):
                try:
                    summary = self.summarize_chunk(chunk, chunk_info)
                    chunk_summaries.append((chunk_info, summary))
                    
                    # Progressive delay to avoid rate limits
                    if i < 10:
                        time.sleep(2)  # 2 seconds for first 10 chunks
                    elif i < 20:
                        time.sleep(4)  # 4 seconds for next 10 chunks
                    else:
                        time.sleep(6)  # 6 seconds for remaining chunks
                        
                except Exception as e:
                    if "429" in str(e):
                        print(f"⏸️  Rate limit hit on chunk {chunk_info['chunk_number']}. Waiting 60 seconds...")
                        time.sleep(60)
                        try:
                            summary = self.summarize_chunk(chunk, chunk_info)
                            chunk_summaries.append((chunk_info, summary))
                        except Exception as retry_error:
                            error_summary = f"Error processing chunk: {str(retry_error)}"
                            chunk_summaries.append((chunk_info, error_summary))
                    else:
                        error_summary = f"Error processing chunk: {str(e)}"
                        chunk_summaries.append((chunk_info, error_summary))
            
            # Step 4: Create JSON summary
            video_summary = self.create_video_summary_json(video_info, chunk_summaries)
            
            # Save to file
            output_file = self.save_summary_json(video_summary, output_path)
            
            # Clean up
            video.close()
            for chunk, _ in chunks:
                chunk.close()
            
            print("=" * 50)
            print("VIDEO SUMMARIZATION COMPLETED!")
            print(f"Summary saved to: {output_file}")
            print("=" * 50)
            
            return video_summary, output_file
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
    
    def view_summary(self, json_file=None):
        """View video summary in a readable format"""
        # Auto-detect summary file if not provided
        if json_file is None:
            json_files = [f for f in os.listdir('.') if f.endswith('_summary.json')]
            if not json_files:
                print("❌ No summary files found in current directory")
                return
            elif len(json_files) == 1:
                json_file = json_files[0]
            else:
                print("📁 Multiple summary files found:")
                for i, file in enumerate(json_files, 1):
                    print(f"   {i}. {file}")
                try:
                    choice = int(input("\nSelect file number: ")) - 1
                    json_file = json_files[choice]
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    return
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print("=" * 60)
            print("VIDEO SUMMARY REPORT")
            print("=" * 60)
            
            # Video info
            print(f"📹 Video: {summary['video_name']}")
            print(f"⏱️  Duration: {summary['total_duration']:.1f} seconds ({summary['total_duration']//60:.0f}m {summary['total_duration']%60:.0f}s)")
            print(f"📊 Resolution: {summary['size'][0]}x{summary['size'][1]}")
            print(f"🎬 FPS: {summary['fps']}")
            print(f"📦 Total Chunks: {summary['total_chunks']}")
            print(f"📅 Processed: {summary['processing_date'][:19]}")
            print()
            
            # Chunk summaries
            print("📝 CHUNK-BY-CHUNK SUMMARIES:")
            print("-" * 60)
            
            for chunk in summary['chunks']:
                print(f"\n🎯 Chunk {chunk['chunk_number']} ({chunk['timestamp']})")
                print(f"📝 Summary ({chunk['summary_length']} chars):")
                
                summary_text = chunk['summary'].replace('\n\n', '\n').strip()
                if len(summary_text) > 300:
                    summary_text = summary_text[:300] + "..."
                
                print(f"   {summary_text}")
                print("-" * 40)
            
            print("\n" + "=" * 60)
            
        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {json_file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")