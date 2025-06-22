from typing import List, Dict, Optional, Union, Tuple
import streamlit as st
import json

class QueryTransformer:
    """
    Implements multiple query transformation techniques:
    1. RAG-fusion - Generates diverse perspectives of the same query
    2. HyDE - Hypothetical Document Embeddings
    3. Query Decomposition - Splits complex queries into sub-queries
    4. Multi-Query Expansion - Paraphrases the original query
    """
    
    def __init__(self, llm):
        self.llm = llm
        print(self.llm)
        
        # RAG fusion prompt (existing)
        self.rag_fusion_prompt = """Generate 5 different versions of this query that capture different aspects and perspectives. 
        Each version should help find relevant educational video content.
        {history}
        Current Query: {query}
        
        Consider these perspectives:
        1. Basic explanation/definition
        2. Practical applications/examples
        3. Technical details/methodology
        4. Related concepts/prerequisites
        5. Common challenges/misconceptions
        
        Return ONLY the transformed queries, one per line. Do not include numbers, labels, or any other text."""
        
        # HyDE prompt
        self.hyde_prompt = """Generate a detailed, factual document that would be a perfect answer to this query. 
        This document will be used to find similar real documents, so be comprehensive and accurate.
        {history}
        Query: {query}
        
        Write a concise but informative document (2-3 paragraphs) that directly addresses this query."""
        
        # Query decomposition prompt
        self.decomposition_prompt = """Break down this complex query into 3-5 simpler sub-queries that together cover all aspects of the original question.
        {history}
        Complex Query: {query}
        
        Return ONLY the sub-queries, one per line. Each sub-query should focus on a distinct aspect of the original question."""
        
        # Multi-query expansion prompt
        self.expansion_prompt = """Rephrase the following query in 5 different ways while preserving its original meaning.
        {history}
        Original Query: {query}
        
        Return ONLY the rephrased queries, one per line. Do not include numbers, labels or any other text."""
        
        # Summary prompt (existing)
        self.summary_prompt = """Provide a very concise summary (2-3 sentences) of the key technical concepts and topics discussed in this conversation that are relevant for the next query.

        Conversation:
        {conversation}
        
        Return ONLY the summary, focusing on technical concepts."""
    
    def _summarize_older_messages(self, conversation_pairs: List[tuple]) -> str:
        """Create a concise summary of older messages"""
        if not conversation_pairs or not self.llm:
            return ""
            
        # Format conversation for summarization
        conversation = ""
        for user_msg, asst_msg in conversation_pairs:
            conversation += f"Q: {user_msg}\nA: {asst_msg}\n\n"
            
        messages = [
            {
                "role": "system",
                "content": "You are a technical conversation summarizer. Create very concise summaries focusing on key concepts."
            },
            {
                "role": "user",
                "content": self.summary_prompt.format(conversation=conversation)
            }
        ]
        
        try:
            response = self.llm.invoke(messages)
            summary = response.content if hasattr(response, 'content') else str(response)
            print(f"@@summary strip length: {summary.strip()}")
            st.write(f"🔍 Summarizing older messages: {summary.strip()}")
            return summary.strip()
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return ""
    
    def _get_chat_history(self, max_pairs: int = 2) -> str:
        """Get recent chat history with summarization for longer conversations"""
        if "messages" not in st.session_state:
            return "No previous context."
            
        messages = st.session_state.messages
        pairs = []
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 1
        
        if not pairs:
            return "No previous context."
        
        history = ""
        
        # If we have more than max_pairs, summarize older conversations
        if len(pairs) > max_pairs:
            older_pairs = pairs[:-max_pairs]
            summary = self._summarize_older_messages(older_pairs)
            if summary:
                history = "Previous discussion summary:\n" + summary + "\n\n"
            
        # Add the most recent conversations in full
        history += "Recent messages:\n"
        recent_pairs = pairs[-max_pairs:]
        for user_msg, assistant_msg in recent_pairs:
            # Truncate very long messages
            user_msg_short = user_msg[:500] + "..." if len(user_msg) > 500 else user_msg
            asst_msg_short = assistant_msg[:500] + "..." if len(assistant_msg) > 500 else assistant_msg
            history += f"User: {user_msg_short}\nAssistant: {asst_msg_short}\n\n"
        
        return history
    
    def _clean_query(self, query: str) -> str:
        """Clean up query by removing leading numbers, dots, and extra whitespace"""
        # Remove leading numbers and dots (e.g., "1.", "2.", etc.)
        cleaned = query.strip()
        while cleaned and cleaned[0].isdigit():
            cleaned = cleaned[1:].strip()
        if cleaned.startswith('.'):
            cleaned = cleaned[1:].strip()
        return cleaned

    def _generate_diverse_queries(self, query: str) -> List[str]:
        """Generate diverse query variations using the LLM"""
        try:
            if not self.llm:
                return [query]
                
            # Get recent chat history
            history = self._get_chat_history()
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates diverse search queries for educational video content. Return the queries as a simple list, one per line."
                },
                {
                    "role": "user",
                    "content": self.rag_fusion_prompt.format(history=history, query=query)
                }
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Clean and filter queries
            queries = [
                self._clean_query(q) for q in content.split('\n') 
                if q.strip() and not q.startswith('{') and not q.startswith('}') 
                and not q.lower().startswith('here') and not q.lower().startswith('"queries"')
            ]
            
            # Always include original query if not present
            if query not in queries:
                queries.insert(0, query)
            
            return queries
            
        except Exception as e:
            st.error(f"Error generating diverse queries: {str(e)}")
            return [query]
    
    def _generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical document that answers the query"""
        try:
            if not self.llm:
                return ""
                
            # Get recent chat history
            history = self._get_chat_history()
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert that creates detailed hypothetical documents to answer queries."
                },
                {
                    "role": "user",
                    "content": self.hyde_prompt.format(history=history, query=query)
                }
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            return content.strip()
            
        except Exception as e:
            st.error(f"Error generating HyDE document: {str(e)}")
            return ""
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into multiple simpler sub-queries"""
        try:
            if not self.llm:
                return [query]
                
            # Get recent chat history
            history = self._get_chat_history()
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at breaking down complex questions into simpler components."
                },
                {
                    "role": "user",
                    "content": self.decomposition_prompt.format(history=history, query=query)
                }
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Clean and filter sub-queries
            sub_queries = [
                self._clean_query(q) for q in content.split('\n') 
                if q.strip() and not q.startswith('{') and not q.startswith('}')
                and not q.lower().startswith('here') and not q.lower().startswith('"sub-queries"')
            ]
            
            # Always include original query if requested subqueries are empty
            if not sub_queries:
                return [query]
            
            return sub_queries
            
        except Exception as e:
            st.error(f"Error decomposing query: {str(e)}")
            return [query]
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate multiple paraphrases of the original query"""
        try:
            if not self.llm:
                return [query]
                
            # Get recent chat history
            history = self._get_chat_history()
                
            # Format prompt for the LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at paraphrasing questions while preserving their meaning."
                },
                {
                    "role": "user",
                    "content": self.expansion_prompt.format(history=history, query=query)
                }
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Clean and filter paraphrased queries
            paraphrased_queries = [
                self._clean_query(q) for q in content.split('\n') 
                if q.strip() and not q.startswith('{') and not q.startswith('}')
                and not q.lower().startswith('here') and not q.lower().startswith('"queries"')
            ]
            
            # Always include original query
            if query not in paraphrased_queries:
                paraphrased_queries.insert(0, query)
            
            return paraphrased_queries
            
        except Exception as e:
            st.error(f"Error expanding query: {str(e)}")
            return [query]
    
    def transform_query(self, user_query: str, method: str = "rag_fusion") -> Union[List[str], Tuple[str, str]]:
        """
        Transform the user query using the specified method.
        
        Args:
            user_query: The original user query
            method: The transformation method to use
                - "rag_fusion": Generate diverse perspectives of the query
                - "hyde": Generate a hypothetical document/answer
                - "decomposition": Break down complex query into sub-queries
                - "expansion": Generate paraphrased variations of the query
                
        Returns:
            - For RAG-fusion, Decomposition, Expansion: List of query variations
            - For HyDE: Tuple of (original_query, hypothetical_document)
        """
        if method == "rag_fusion":
            st.write("🔄 Applying RAG-fusion to generate diverse queries...")
            queries = self._generate_diverse_queries(user_query)
            if len(queries) > 1:
                st.write("Generated query variations:")
                print("Generated query variations:")
            else:
                st.warning("Could not generate diverse queries, using original query only.")
                print("Using original query only.")
            return queries
            
        elif method == "hyde":
            st.write("🔄 Applying HyDE (Hypothetical Document Embeddings)...")
            hypothetical_doc = self._generate_hyde_document(user_query)
            if hypothetical_doc:
                st.write("Generated hypothetical document for retrieval")
                print("Generated hypothetical document for retrieval")
            else:
                st.warning("Could not generate hypothetical document, using original query only.")
                print("Failed to generate hypothetical document")
            return (user_query, hypothetical_doc)
            
        elif method == "decomposition":
            st.write("🔄 Applying Query Decomposition...")
            sub_queries = self._decompose_query(user_query)
            if len(sub_queries) > 1:
                st.write(f"Decomposed query into {len(sub_queries)} sub-queries")
                print(f"Decomposed query into {len(sub_queries)} sub-queries")
            else:
                st.warning("Could not decompose query, using original query only.")
                print("Using original query only (decomposition failed)")
            return sub_queries
            
        elif method == "expansion":
            st.write("🔄 Applying Multi-Query Expansion...")
            expanded_queries = self._expand_query(user_query)
            if len(expanded_queries) > 1:
                st.write(f"Generated {len(expanded_queries)} query paraphrases")
                print(f"Generated {len(expanded_queries)} query paraphrases")
            else:
                st.warning("Could not generate query paraphrases, using original query only.")
                print("Using original query only (expansion failed)")
            return expanded_queries
            
        else:
            st.warning(f"Unknown transformation method: {method}, using original query")
            print(f"Unknown transformation method: {method}, using original query")
            return [user_query]