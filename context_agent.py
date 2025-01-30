from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.types import Command
from typing import List, Dict, Literal
from pydantic import BaseModel
import numpy as np
from typing import Tuple,Annotated,List
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer   
import bm25s
from termcolor import colored 

#model = SentenceTransformer('intfloat/e5-large-v2')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#model = SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1')
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
class ContextScore(BaseModel):
    score: float

def get_similarity_score(user_request: str, context: str) -> float:
    embeddings = model.encode([user_request, context])
    similarity_score = np.dot(embeddings[0], embeddings[1]) #/ (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return similarity_score

def calculate_context_relevance(user_request: str, context: str, llm) -> float:
    """Calculate relevance score between user request and context"""
    prompt = f"""
    Analyze the relevance between the user request and the given context.
    User Request: {user_request}
    Context: {context}
    
    Rate the relevance from 0.0 to 1.0, where:
    - 1.0: Highly relevant and directly addresses the request
    - 0.0: Completely irrelevant
    
    Return only the numerical score.
    """
    parser = PydanticOutputParser(pydantic_object=ContextScore)
    format_instructions = parser.get_format_instructions()
    prompt = ("user",f"Analyze the relevance between the user request and the given context. \n User Request: {user_request} \n Context: {context} \n {format_instructions}")
    chain = llm.invoke(prompt)
    response = parser.parse(chain)
    try:
        score = float(response.score)
        return min(max(score, 0.0), 1.0)
    except:
        return 0.0

def get_highest_relevance_messages(
    user_request: str, 
    state: dict, 
    llm, 
    agent_keys: List[str] = ["writer", "critic", "artist", "genral_assistance_"]
) -> Tuple[List[BaseMessage]]:
    """
    Calculates the relevance score for each agent's messages and returns the messages
    from the agent with the highest score.

    Args:
        user_request: The current user request.
        state: The current state dictionary.
        llm: The language model.
        agent_keys: List of agent keys to consider.

    Returns:l
        A tuple containing:
          - The list of messages with the highest relevance score.
          - The name of the agent with the highest relevance score.
    """
    highest_score = 0
    highest_score_messages = []
    highest_score_agent = None
    print(colored(f"\n----- {user_request} -----\n", "yellow","on_grey"))
    for agent_key in agent_keys:
        agent_messages = state.get(agent_key, [])
        if agent_messages:
            #print(f"agent_key: {agent_key}")
            scored_messages = []
            courpous = []
            # Iterate through messages in chunks of 3
            for i in range(0, len(agent_messages)):
                message_chunk = agent_messages[i].content
                courpous.append(message_chunk)
                score = get_similarity_score(user_request, message_chunk)
                if score > 0.3:
                    print(f"score: {score} {message_chunk}")
                    scored_messages.append((score, message_chunk))
            # scored_messages.sort(key=lambda x: x[0], reverse=True)
            #print(f"scored_messages: {scored_messages}") 
            corpus_tokens = bm25s.tokenize(courpous)
            retriever = bm25s.BM25(corpus=courpous)
            retriever.index(corpus_tokens)   
            query_tokens = bm25s.tokenize(user_request)
            docs, scores = retriever.retrieve(query_tokens, k=min(10,len(agent_messages)))
            docs = docs[0].tolist()
            top_messages_unsorted = []
            for _, msgs in scored_messages[:len(scored_messages)+1]:
                if msgs not in docs:
                    top_messages_unsorted.append(msgs)
            
            #print(f"top_messages_unsorted: {top_messages_unsorted}")
            top_messages_unsorted.extend(docs)
            top_messages = []
            for msg in agent_messages:
                if msg.content in top_messages_unsorted:
                    top_messages.append(msg)
            #print(f"top_messages: {top_messages}")
            
            # Ensure we always include the last 3 messages
            last_3_messages = agent_messages[-3:]
            
            # Filter out the last 3 messages from top_messages to avoid duplicates
            top_messages = [msg for msg in top_messages if msg not in last_3_messages]
            
            # Combine the top messages with the last 3 messages
            top_messages = top_messages + last_3_messages
            
            if len(scored_messages)>0 :
                highest_score = scored_messages[0][0]
                highest_score_messages = top_messages
                highest_score_agent = agent_key

    return highest_score_messages
