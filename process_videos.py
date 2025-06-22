import os
import time
from rag_pipeline.video_summarizer_Gemini_Chunk import VideoSummarizer
# from rag_pipeline.video_summarizer_OpenAI_Frame import VideoSummarizer
from rag_pipeline.video_embedder import VideoEmbedder

def interactive_mode():
    print("Video RAG System - Process Videos")
    print("=" * 50)
    print("Automatically handles videos in ANY language")
    
    summarizer = VideoSummarizer()
    embedder = VideoEmbedder(model_name="all-MiniLM-L6-v2")
    # embedder.delete_collection(collection_name="video_summaries")
    
    while True:
        videos_dir = "videos"
        if not os.path.exists(videos_dir):
            print(f"Videos folder not found: {videos_dir}")
            break
        
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"No video files found in {videos_dir}/")
            break
        
        print("\nAvailable videos:")
        for i, video in enumerate(video_files, 1):
            print(f"   {i}. {video}")
        
        print(f"   {len(video_files) + 1}. Process ALL videos (batch)")
        print(f"   {len(video_files) + 2}. Exit")
        
        try:
            choice = input(f"\nSelect option (1-{len(video_files) + 2}): ").strip()
            
            if choice == str(len(video_files) + 2) or choice.lower() == 'exit':
                print("\nBye!")
                break
            elif choice == str(len(video_files) + 1):
                print("\nProcessing ALL videos...")
                successful_processes = 0
                failed_processes = []
                start_time = time.time()
                
                for i, video_file in enumerate(video_files, 1):
                    video_path = os.path.join(videos_dir, video_file)
                    
                    print(f"Processing {i}/{len(video_files)}: {video_file}")
                    print(f"{'='*60}")
                    
                    try:
                        summary, output_path = summarizer.process_video(video_path)
                        
                        print(f"Summary created: {output_path}")
                        embedded_summary, collection = embedder.process_summary_json(output_path)
                        
                        print(f"Embeddings collection created: {collection.name}")
                        
                        successful_processes += 1
                        
                    except Exception as e:
                        print(f"Error processing {video_file}: {str(e)}")
                        failed_processes.append((video_file, str(e)))
                    
                    # Show progress
                    progress = (i / len(video_files)) * 100
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {i}/{len(video_files)} ({progress:.1f}%)")
                
                # Final summary
                total_time = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"BATCH PROCESSING COMPLETE!")
                print(f"Successfully processed: {successful_processes}/{len(video_files)} videos")
                print(f"Total time: {total_time/60:.1f} minutes")
                
                if failed_processes:
                    print(f"\nFailed processes ({len(failed_processes)}):")
                    for video, error in failed_processes:
                        print(f"   • {video}: {error}")
                
                break 
            else:
                video_choice = int(choice) - 1
                if video_choice < 0 or video_choice >= len(video_files):
                    print("Invalid selection")
                    continue
                selected_video = video_files[video_choice]
                video_path = os.path.join(videos_dir, selected_video)
                
                print(f"\nProcessing: {selected_video}")
                print("Any language will be automatically translated to English")
                
                try:
                    summary, output_path = summarizer.process_video(video_path)
                    
                    print(f"Summary created: {output_path}")
                    embedded_summary, collection = embedder.process_summary_json(output_path)
                    print(f"Embeddings collection created : {collection.name}")
                    
                except Exception as e:
                    print(f"Error processing {selected_video}: {str(e)}")

                another = input("\nProcess another video? (y/n): ").lower().strip()
                if another not in ['y', 'yes', '']:
                    print("\nBye!")
                    break
        
        except ValueError:
            print("Invalid selection. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == "view":
            summarizer = VideoSummarizer()
            # summarizer.view_summary()
        elif sys.argv[1] == "process" and len(sys.argv) > 2:
            video_path = sys.argv[2]
            summarizer = VideoSummarizer()
            summarizer.process_video(video_path)
        else:
            print("Usage:")
            print("  python process_videos.py                    # Interactive mode")
            print("  python process_videos.py view               # View summaries")
            print("  python process_videos.py process <video>    # Process specific video")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()