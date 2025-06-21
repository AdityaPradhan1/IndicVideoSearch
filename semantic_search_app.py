import streamlit as st
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# model options for the semantic search tool
NATIVE_MODELS = {
    "Muril": "google/muril-base-cased",
    "IndicSBERT": "l3cube-pune/indic-sentence-bert-nli",
    "LaBSE": "sentence-transformers/LaBSE",
    "IndicBERT": "ai4bharat/IndicBERTv2-MLM-only",
    "Krutim Vyakyarth": "krutrim-ai-labs/Vyakyarth",
    "Roberta-base":"FacebookAI/xlm-roberta-base",
    "Rembert (google)":"google/rembert"
}
ENGLISH_MODELS = {
    "MiniLM (default)": "sentence-transformers/all-MiniLM-L6-v2",
    "BERT-base-nli": "sentence-transformers/bert-base-nli-mean-tokens",
    "DistilRoBERTa": "sentence-transformers/all-distilroberta-v1",
    "Mpnet":"sentence-transformers/all-mpnet-base-v2"
}

@st.cache_resource
def load_models(native_name, english_name):
    native_model = SentenceTransformer(NATIVE_MODELS[native_name])
    english_model = SentenceTransformer(ENGLISH_MODELS[english_name])
    translator = GoogleTranslator(source='auto', target='en') 
    return native_model, english_model, translator

def render_app():
    st.sidebar.header("Model Selection")
    selected_native_model = st.sidebar.selectbox("Native Language Model", list(NATIVE_MODELS.keys()))
    selected_english_model = st.sidebar.selectbox("English Embedding Model", list(ENGLISH_MODELS.keys()))

    native_model, english_model, translator = load_models(selected_native_model, selected_english_model)

    st.title("Semantic Search: Native Embedding vs. Translated Embedding")
    st.markdown("""
    Compare semantic similarity between a **query** and **corpus** entries using two methods:
    1.  **Native Embedding**: Directly embedding text in its original Indian language.
    2.  **Translated Embedding**: Translating text to English before embedding.
    """)

    query = st.text_input("Enter your query (in Hindi, Tamil, Malayalam, etc.): eg मैं सब्ज़ी लेने जा रहा हूँ")
    corpus_input = st.text_area(
        "Enter corpus (one sentence per line, in the same language):",
        "दिल्ली जाने वाली अगली ट्रेन शाम को 6 बजे है।\nநான் டெல்லிக்கு செல்வதற்கான ரயிலை தேடுகிறேன்.\nഞാൻ ഡൽഹിയിലേക്ക് പോകുന്ന ട്രെയിൻ എപ്പോഴാണെന്ന് അറിയണം.\nআমি জানতে চাই যে কখন দিল্লির ট্রেন ছাড়বে।\nमी दिल्लीसाठी ट्रेन केव्हा आहे ते पाहतो आहे.\nನಾನು ದೆಹಲಿಗೆ ಹೋಗುವ ರೈಲು ಯಾವಾಗ ಬರುತ್ತದೆ ಎಂದು ತಿಳಿದುಕೊಳ್ಳಲು ಬಯಸುತ್ತೇನೆ.\nనేను ఢిల్లీకి వెళ్లే రైలు ఎప్పుడుందో తెలుసుకోవాలి.\nહું દિલ્હીની ટ્રેન ક્યારે છે તે જાણવા માંગું છું.\nI need to find the next train to Delhi.\nIs there a bus from Jaipur to Delhi tonight?\nMy flight to Delhi has been delayed.\nWe missed our train and had to book a cab.\nHe is traveling to Delhi by car tomorrow.\nI want to book a train ticket for next weekend.\nThere is heavy traffic on the highway to Delhi.\nThe station is crowded today because of the holiday.\nOur bus to Agra leaves in 30 minutes.\nWe are taking a flight from Mumbai to Delhi.\nThey are checking ID cards at the railway station.\nShe prefers to travel by train rather than bus.\nCan I get a sleeper seat on the Delhi express?\nI reached Delhi early in the morning.\nमैं बाज़ार जा रहा हूँ\nഞാൻ കടയിലേക്ക് പോകുന്നു\nநான் கடைக்கு போகிறேன்\nमैं आज ऑफिस नहीं जा पा रहा हूँ।\nनौकरी के लिए इंटरव्यू कल सुबह है।\nപഠനത്തിന് ഏറ്റവും നല്ല സർവകലാശാല ഏതാണ്?\nநான் வேலைக்கு விண்ணப்பித்துள்ளேன்.\nআমি আগামী সপ্তাহে পরীক্ষা দিচ্ছি।\nಮಗನು ಇಂದು ಶಾಲೆಗೆ ಹೋಗಿಲ್ಲ.\nనా కోర్సు పూర్తవడానికి ఇంకా ఆరు ತಿಂాలు మిగిలున్నాయి.\nહું આજે વર્ક ફ્રોમ હોમ છું.\nI'm working on a big project for my company.\nShe is studying data science online.\nWe had a team meeting this morning.\nI need to submit my assignment by tonight.\nThey are looking for a new job in Bangalore.\nHe didn't attend the lecture yesterday.\nClasses will resume after the festival holidays.\nCan you help me prepare for the exam?\nThis training program is very helpful.\nThey just started their internship.\nMy professor gave feedback on my report.\nDo you want to join this online course?\nHe studying for a master's at IISC.\nI forgot to bring my notebook to class.\nWe have a presentation on Monday.\nMy school is conducting a science exhibition.\nThe teacher explained the topic very well.\nThe exam schedule was released today.\nI will start my job from next month.\nShe wants to study medicine.\nI prefer studying late at night.\nHis office is in Hyderabad.\nMy college is offering a new AI course.\nमैं सब्ज़ी मंडी जा रहा हूँ।\nநான் காய்கறி கடைக்குப் போகிறேன்.\nഞാൻ പച്ചക്കറി മാർക്കറ്റിലേക്ക് പോകുന്നു.\nআমি বাজারে সবজি কিনতে যাচ্ছি।\nमी भाजीसाठी बाजारात जात आहे.\nನಾನು ತರಕಾರಿ ಖರೀದಿಸಲು ಹೊರಟಿದ್ದೇನೆ.\nనేను కూరగాయల కోసం మార్కెట్కి వెళ్తున్నాను.\nહું બજારમાંથી શાકભાજી લાવવા જઈ રહ્યો છું.\nI'm going to the grocery store to buy vegetables.\nWe are planning to cook tonight and need fresh vegetables.\nI just bought some tomatoes and spinach.\nThe market is crowded today because of the festival.\nMy sister is preparing dinner and asked me to get some veggies.\nHe went out to get fruits, not vegetables.\nI'm visiting the store next to the vegetable stall.\nShe is looking for organic produce.\nI forgot to buy onions and potatoes.\nThe prices of vegetables have gone up this week.\nHe is a regular customer at the vegetable shop.\nThey are selling farm-fresh vegetables today.\nI love shopping for vegetables early in the morning.\n"
    )

    if st.button("Run Comparison"):
        if not query or not corpus_input:
            st.warning("Please enter both a query and a corpus.")
            return

        corpus = corpus_input.strip().split("\n")

        # --- Native Language Embedding ---
        st.markdown("### Native Language Embedding Result:")
        with st.spinner("Running native language embedding..."):
            chroma_native = chromadb.Client(Settings(anonymized_telemetry=False))
            embedding_fn_native = SentenceTransformerEmbeddingFunction(model_name=NATIVE_MODELS[selected_native_model])
            if "native" in [c.name for c in chroma_native.list_collections()]:
                chroma_native.delete_collection("native")
            collection_native = chroma_native.create_collection(name="native", embedding_function=embedding_fn_native)
            collection_native.add(documents=corpus, ids=[f"native_{i}" for i in range(len(corpus))])
            results_native = collection_native.query(query_texts=[query], n_results=len(corpus))
            for rank, (doc, dist) in enumerate(zip(results_native['documents'][0], results_native['distances'][0])):
                st.write(f"{rank+1}. {doc} (distance: {dist:.4f})")

        st.markdown("---")

        st.markdown("### Translated + English Embedding Result:")
        with st.spinner("Translating and running English embedding..."):
            try:
                def safe_translate(text):
                    result = translator.translate(text)
                    return result

                corpus_en = [safe_translate(text) for text in corpus]
                query_en = safe_translate(query)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                corpus_en, query_en = [], ""

            if query_en:
                st.write(f"Query translated to: '{query_en}'")
                chroma_en = chromadb.Client(Settings(anonymized_telemetry=False))
                embedding_fn_en = SentenceTransformerEmbeddingFunction(model_name=ENGLISH_MODELS[selected_english_model])
                if "translated" in [c.name for c in chroma_en.list_collections()]:
                    chroma_en.delete_collection("translated")
                collection_en = chroma_en.create_collection(name="translated", embedding_function=embedding_fn_en)
                collection_en.add(documents=corpus_en, ids=[f"en_{i}" for i in range(len(corpus_en))])
                results_en = collection_en.query(query_texts=[query_en], n_results=len(corpus_en))
                for rank, (doc, dist) in enumerate(zip(results_en['documents'][0], results_en['distances'][0])):
                    original_doc = corpus[corpus_en.index(doc)]
                    st.write(f"{rank+1}. {original_doc} (translated: '{doc}') (distance: {dist:.4f})")

if __name__ == "__main__":
    render_app()