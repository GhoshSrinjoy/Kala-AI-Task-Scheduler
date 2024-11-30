import streamlit as st
import chromadb
from datetime import datetime, timedelta
import pandas as pd
from streamlit_calendar import calendar as st_calendar
import json
import time
from typing import Dict, List
import requests
import hashlib

# Configure page settings
st.set_page_config(
    layout="wide",
    page_title="Calendar Assistant",
    page_icon="📅",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    .stSpinner > div {
        text-align: center;
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #2e3136;
    }
    .assistant-message {
        background-color: #1e1e1e;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    client = chromadb.Client()
    collection = client.create_collection(
        name="calendar_events",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# Cache for Ollama embeddings
@st.cache_data(ttl=3600)
def get_embedding(_text: str) -> List[float]:
    try:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": _text
            }
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            st.error("Failed to generate embedding. Please check if Ollama is running.")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama. Make sure it's running on port 11434")
        return None
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def get_llm_response(query: str, context: Dict = None) -> str:
    try:
        messages = []
        
        # System message with context
        system_msg = """You are a helpful calendar assistant. You help users find and understand their calendar events.
        Be concise and friendly in your responses. If no relevant events are found, let the user know politely."""
        
        messages.append({"role": "system", "content": system_msg})
        
        # Add context if available
        if context and context['documents'] and len(context['documents'][0]) > 0:
            context_msg = "Here are the relevant calendar events:\n\n"
            for doc, meta in zip(context['documents'][0], context['metadatas'][0]):
                context_msg += f"Event: {meta['title']}\nWhen: {meta['start']}\nDetails: {doc}\n\n"
            messages.append({"role": "system", "content": context_msg})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Get response from LLM
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.1",
                "messages": messages,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return "I apologize, but I'm having trouble processing your request right now."
            
    except Exception as e:
        return f"I apologize, but there was an error processing your request: {str(e)}"

# Cache calendar data
@st.cache_data(ttl=60)
def load_calendar_data() -> List[Dict]:
    try:
        collection = init_chromadb()
        results = collection.get()
        events = []
        for idx, (meta, text) in enumerate(zip(results['metadatas'], results['documents'])):
            event = {
                "id": str(idx),
                "title": meta.get('title', 'Untitled'),
                "start": meta.get('start'),
                "end": meta.get('end'),
                "description": text,
                "backgroundColor": meta.get('color', '#1976D2')
            }
            events.append(event)
        return events
    except Exception as e:
        st.error(f"Error loading calendar data: {e}")
        return []

def add_event(title: str, start: str, end: str, description: str, color: str):
    collection = init_chromadb()
    
    event_id = hashlib.md5(f"{title}{start}{end}".encode()).hexdigest()
    event_text = f"{title} {description}"
    embedding = get_embedding(event_text)
    
    if embedding is None:
        st.error("Failed to generate embedding for event. Please try again.")
        return False
        
    try:
        collection.add(
            documents=[description],
            embeddings=[embedding],
            metadatas=[{
                "title": title,
                "start": start,
                "end": end,
                "color": color,
                "event_id": event_id
            }],
            ids=[event_id]
        )
        
        load_calendar_data.clear()
        return True
    except Exception as e:
        st.error(f"Error adding event: {str(e)}")
        return False

def semantic_search(query: str, n_results: int = 5):
    try:
        collection = init_chromadb()
        if collection.count() == 0:
            st.info("No events found in calendar. Add some events first!")
            return {"documents": [], "metadatas": []}
            
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return {"documents": [], "metadatas": []}
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count())
        )
        return results
    except Exception as e:
        st.error(f"Error searching calendar: {str(e)}")
        return {"documents": [], "metadatas": []}

def main():
    st.title("📅 Smart Calendar")
    
    # Initialize session states
    if 'calendar_view' not in st.session_state:
        st.session_state.calendar_view = 'dayGridMonth'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create columns for view buttons
    cols = st.columns(4)
    
    # View buttons with active state
    button_style = """
        <style>
            div[data-testid="stHorizontalBlock"] button {
                background-color: #262730;
                color: white;
                border: 1px solid #4B4B4B;
            }
            div[data-testid="stHorizontalBlock"] button:hover {
                background-color: #0E1117;
                border-color: #00BFFF;
            }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    if cols[0].button('📅 Month', use_container_width=True):
        st.session_state.calendar_view = 'dayGridMonth'
    if cols[1].button('📆 Week', use_container_width=True):
        st.session_state.calendar_view = 'timeGridWeek'
    if cols[2].button('📋 Day', use_container_width=True):
        st.session_state.calendar_view = 'timeGridDay'
    if cols[3].button('📝 List', use_container_width=True):
        st.session_state.calendar_view = 'listWeek'
    
    # Split main content area
    col1, col2 = st.columns([7, 3])
    
    with col1:
        events_data = load_calendar_data()
        calendar_options = {
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": ""
            },
            "initialView": st.session_state.calendar_view,
            "selectable": True,
            "navLinks": True,
            "editable": True,
            "dayMaxEvents": True,
            "height": 600,
            "slotMinTime": "06:00:00",
            "slotMaxTime": "22:00:00",
            "eventTimeFormat": {
                "hour": "2-digit",
                "minute": "2-digit",
                "meridiem": False,
                "hour12": False
            }
        }
        
        st_calendar(events=events_data, options=calendar_options)
    
    with col2:
        st.subheader("Add Event")
        with st.form("event_form", clear_on_submit=True):
            title = st.text_input("Event Title")
            col_start, col_end = st.columns(2)
            
            with col_start:
                start_date = st.date_input("Start Date")
                start_time = st.time_input("Start Time")
            with col_end:
                end_date = st.date_input("End Date")
                end_time = st.time_input("End Time")
            
            description = st.text_area("Description")
            color = st.color_picker("Event Color", "#1976D2")
            
            submit_button = st.form_submit_button("Add Event", use_container_width=True)
            
            if submit_button:
                if not title:
                    st.error("Please enter an event title")
                    return
                
                if end_date < start_date:
                    st.error("End date cannot be before start date")
                    return
                
                start = f"{start_date}T{start_time}"
                end = f"{end_date}T{end_time}"
                
                with st.spinner("Adding event..."):
                    if add_event(title, start, end, description, color):
                        st.success("Event added successfully!")
                        time.sleep(1)
                        st.rerun()
    
    # Sidebar with chat interface
    with st.sidebar:
        st.title("💬 Calendar Assistant")
        query = st.text_input("Ask about your calendar...")
        
        if query:
            with st.spinner("Processing your query..."):
                # Get relevant events through semantic search
                search_results = semantic_search(query)
                
                # Get LLM response with context
                llm_response = get_llm_response(query, search_results)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>You:</b> {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <b>Assistant:</b> {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()