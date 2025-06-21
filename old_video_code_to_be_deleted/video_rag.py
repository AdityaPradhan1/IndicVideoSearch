#!/usr/bin/env python3
"""
Video RAG System - All-in-One Video Summarizer
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
import json
import os
import numpy as np
from langchain_community.vectorstores import Chroma

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VideoSummarizer:
    def __init__(self):
        # Configure Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Configuration
        self.chunk_duration = 30  # 10 seconds per chunk
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
        """Function 2: Segment video into 10-second chunks"""
        print("Segmenting video into 10-second chunks...")
        
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
        """Function 3: Summarize video chunk using Gemini 2.5 Pro"""
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
    
    def embed_summaries(self, video_summary):
        """Function 5: Create embeddings for summaries"""
        print("Creating embeddings for summaries...")
        
        full_text = f"Video: {video_summary['video_name']}\n"
        
        for chunk in video_summary['chunks']:
            full_text += f"Time {chunk['timestamp']}: {chunk['summary']}\n"
        
        video_summary['embedding_info'] = {
            'full_text': full_text,
            'text_length': len(full_text),
            'embedding_ready': True,
            'embedding_date': datetime.now().isoformat()
        }
        
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
                                # Update the existing summary
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
            
            # Re-create embeddings
            existing_summary = self.embed_summaries(existing_summary)
            
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
        """Main function to process entire video"""
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
                    
                    user_choice = input("💡 Summary already exists. Use existing? (y/n/view): ").lower().strip()
                    
                    if user_choice == 'y' or user_choice == 'yes' or user_choice == '':
                        print("✅ Using existing summary (no API calls needed)")
                        print("=" * 50)
                        print("VIDEO SUMMARIZATION COMPLETED!")
                        print(f"Summary loaded from: {existing_path}")
                        print("=" * 50)
                        return existing_summary, existing_path
                    
                    elif user_choice == 'view' or user_choice == 'v':
                        print("\n📖 Showing existing summary:")
                        self.view_summary(existing_path)
                        
                        reuse_choice = input("\n💡 Use this existing summary? (y/n): ").lower().strip()
                        if reuse_choice == 'y' or reuse_choice == 'yes' or reuse_choice == '':
                            print("✅ Using existing summary")
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
            
            # Step 3: Summarize each chunk with better rate limiting
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
            
            # Step 5: Add embeddings
            video_summary = self.embed_summaries(video_summary)
            
            # Save to file
            output_file = self.save_summary_json(video_summary, output_path)
            
            # Vectorize and persist in ChromaDB
            self.vectorize_summary_json(output_file)
            
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
            
            # Embedding info
            print(f"\n🔍 EMBEDDING INFO:")
            print(f"   Total text length: {summary['embedding_info']['text_length']} characters")
            print(f"   Ready for search: {summary['embedding_info']['embedding_ready']}")
            print("\n" + "=" * 60)
            
        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {json_file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

    def vectorize_summary_json(self, summary_json_path, persist_directory="chroma_db"):
        """Load summary JSON, vectorize only the summary fields, and store in Chroma DB with metadata."""
        print(f"Vectorizing summary from: {summary_json_path}")
        
        # Load summary data from JSON
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
        # Initialize Chroma client
        client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
        
        # Delete existing collection if it exists
        try:
            client.delete_collection("video_summaries")
        except:
            pass
            
        # Create new collection
        collection = client.create_collection(
            name="video_summaries",
            metadata={"description": "Video chunk summaries with temporal information"}
        )
        
        # Initialize the embedding model
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Add documents to collection
        documents = []
        metadatas = []
        ids = []
        
        for chunk in summary_data["chunks"]:
            summary_text = chunk["summary"]
            metadata = {
                "chunk_number": str(chunk["chunk_number"]),  # Convert to strings for ChromaDB
                "start_time": str(chunk["start_time"]),
                "end_time": str(chunk["end_time"]),
                "video_name": summary_data["video_name"],
                "video_path": summary_data["video_path"]
            }
            doc_id = f"{summary_data['video_name']}_chunk_{chunk['chunk_number']}"
            
            documents.append(summary_text)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        # Add all documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ All summaries vectorized and stored in ChromaDB at {persist_directory}!")
        return collection

def interactive_mode():
    """Interactive mode for processing videos"""
    print("🎬 Video RAG System - Process Videos")
    print("=" * 50)
    print("✅ Automatically handles videos in ANY language")
    print("✅ Gemini translates everything to English")
    print()
    
    summarizer = VideoSummarizer()
    
    while True:
        # Check for videos directly
        videos_dir = "videos"
        if not os.path.exists(videos_dir):
            print(f"❌ Videos folder not found: {videos_dir}")
            break
        
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"❌ No video files found in {videos_dir}/")
            break
        
        print("\n📁 Available videos:")
        for i, video in enumerate(video_files, 1):
            print(f"   {i}. {video}")
        
        print(f"   {len(video_files) + 1}. Exit")
        
        try:
            choice = input(f"\nSelect video (1-{len(video_files) + 1}): ").strip()
            
            if choice == str(len(video_files) + 1) or choice.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            video_choice = int(choice) - 1
            if video_choice < 0 or video_choice >= len(video_files):
                print("❌ Invalid selection")
                continue
                
            selected_video = video_files[video_choice]
            video_path = os.path.join(videos_dir, selected_video)
            
            print(f"\n🎬 Processing: {selected_video}")
            print("🌍 Any language will be automatically translated to English")
            
            summary, output_path = summarizer.process_video(video_path)
            
            print(f"\n✅ Success! Summary saved to: {output_path}")
            print("\n" + "=" * 50)
            
        except ValueError:
            print("❌ Invalid selection")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            
        # Ask if user wants to process another video
        another = input("\n🔄 Process another video? (y/n): ").lower().strip()
        if another not in ['y', 'yes', '']:
            print("\n👋 Goodbye!")
            break

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == "view":
            summarizer = VideoSummarizer()
            summarizer.view_summary()
        elif sys.argv[1] == "process" and len(sys.argv) > 2:
            video_path = sys.argv[2]
            summarizer = VideoSummarizer()
            summarizer.process_video(video_path)
        else:
            print("Usage:")
            print("  python video_rag.py                    # Interactive mode")
            print("  python video_rag.py view               # View summaries")
            print("  python video_rag.py process <video>    # Process specific video")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()

