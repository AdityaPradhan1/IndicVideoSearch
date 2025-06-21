# Environment Setup

 - Please install Python 3.12 if not available

 - Create a Virtual Env in the root directory
 ```bash
 python3.12 -m venv searchenv
 source searchenv/bin/activate
 ```

- Add all relevant API keys
```bash
export SARVAMAI_API_KEY="<YOUR-KEY>"
export GEMINI_API_KEY="<YOUR-KEY>"
export GOOGLE_API_KEY="<SAME KEY>" #Code is referencing it in 2 different ways. needs to be fixed by relevant team member
```
 - Install libraries
 ```bash
 sudo apt install ffmpeg
 pip3 install -r requirements.txt
 ```

- Process Some videos first. You will need the GEMINI_API_KEY to do this step. Otherwise you can use the already available chroma db in the chroma_db folder.
```bash
# There is a sample video in the videos folder. You can add more videos here. THen run
rm -rf chroma_db/chroma.sqlite3 #Optional step. There is a preprocessed db here. You can start fresh if you delete this d

python3 process_videos.py #Follow the interactive mode
```

 - Run the streamlit app
 ```bash
 streamlit run main_app.py
 (or)
 streamlit run main_app.py --server.fileWatcherType none #If you see some torch related errors in the terminal. You will have to manually restart the streamlit app from terminal after every file change in this case as the run & rerun commands won't be available
 ```

# Notes regarding modules and team member contribution
The rag_pipeline folder has code regarding Video Summarizing, Embedding and Vector Search - Aju, Aditya and Keshav will need to look at these files

The chat_app folder has code regarding Audio Processing, Query Transformation and RAG Interfaces - Karthik, Srishti and Keshav need to focus here

Aditya worked on alternative Embedding Models that can be used. We use IndicSBERT in VideoEmbedder. Users can test playing with different embedding models, queries and corpus by selecting semantic search comparison otherwise they can see cmparisons we made to select best pre trained language model by running the jupyter nb.

Currently the query_transformation.py file contains the decomposition & hyde methods Karthik had shared. But I have parameterized it. So in actual execution, those transformations are not called. Srishti can make changes as needed. Query Transformation can also mean doing multiple RAG searches by generating different queries based on user's current question and previous chat context. This is similar to the RAG Fusion concept explained in class. All this are alternatives. We will finally use the alternative Shrishti suggests.

the llm_clients folder now has the sarvam_client. Karthik works on this. Other LLM clients can also be moved here. That is work for the respective team member.

config.py has some configurations including the current chat LLM model apart from Sarvam AI. It is a gemini model. The code in chat_app/message_handler.py will use this llm model. Currently it is specific to Gemini models only as that is how Karthik had implemented it. This can also be modified based on the final chat model Karthik Identifies.

 # [OLD] Video RAG Pipeline
 The video RAG pipeline need not necessarily be part of the chat app. We can preprocess the data, store it in a vector DB and call the search functionality alone. This will keep the chat app separate from the video processing pipeline. RAG Pipeline code can be in separate python classes. We will just import search.

 # [OLD] Video Summary Pipeline
 This will create a summary JSON file for the selected videos. Process the selected video (from the video folder) and Save the summary to the output folder. Each video is segmented into 10-second chunks, with The Video Summary Pipeline processes selected videos to create comprehensive JSON summary files stored in the output folder. Each video is automatically segmented into 10-second chunks, with both audio and visual content analyzed by Gemini AI. These summaries include precise timestamps, visual scene descriptions, spoken content transcriptions, and contextual information, all organized in  JSON format.

 Run Video Summary Pipline:
  ```bash
 python video_rag.py 
 ```     