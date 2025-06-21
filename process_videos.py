import os
from rag_pipeline.video_summarizer import VideoSummarizer
from rag_pipeline.video_embedder import VideoEmbedder

def interactive_mode():
    """Interactive mode for processing videos"""
    print("🎬 Video RAG System - Process Videos")
    print("=" * 50)
    print("✅ Automatically handles videos in ANY language")
    print("✅ Gemini translates everything to English")
    print()
    
    summarizer = VideoSummarizer()
    embedder = VideoEmbedder(model_name="all-MiniLM-L6-v2")
    embedder.delete_collection(collection_name="video_summaries")
    
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
            
            embedded_summary, collection = embedder.process_summary_json(output_path)
            print(f"✅ Summary vectorized and stored in collection: {collection}")
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