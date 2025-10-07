*“My calendar just recommended a meeting with myself. Finally, someone gets me.”*  

# 🧠 Kala AI Task Scheduler  

A modern, **AI-powered calendar** that doesn’t just store your schedule , it *understands* it.  
Built with **Streamlit**, **ChromaDB**, and **LLaMA**, it blends event management with semantic search and natural language interaction.  

🔗 **Repo:** https://github.com/GhoshSrinjoy/Kala-AI-Task-Scheduler  

---

## Overview  

Kala AI is what happens when a traditional calendar meets an AI assistant.  
You can create, manage, and query events , but instead of typing filters and dates, you can just *ask*:  

> “What meetings do I have next week with the design team?”  

It knows. It finds. It tells you.  

---

## ✨ Features  

### 🗓️ Calendar Management  
- Month, Week, Day, and List views  
- Add or delete events easily  
- Drag-and-drop interface  
- Color-coded event types  
- Interactive management panel  

### 🤖 AI Integration  
- Semantic event search using vector embeddings  
- Natural language chat interface  
- Context-aware insights powered by LLaMA  
- Real-time event recommendations  

### ⚙️ Technical Features  
- **ChromaDB** for semantic storage  
- **LLaMA via Ollama** for NLU  
- Real-time updates and caching  
- Responsive **Streamlit** UI  
- Persistent chat history  

## Installation 🚀

1. Install the required dependencies:
```bash
pip install streamlit chromadb pandas streamlit-calendar requests
```

2. Install Ollama and pull required models:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama2
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage 💡

### Adding Events
1. Fill in the event details in the "Add Event" form
2. Provide title, date/time, description, and color
3. Click "Add Event" to save

### Managing Events
1. Click "Manage Events" to see all calendar entries
2. Use the delete button to remove events
3. View events in different calendar formats

### Using AI Assistant
1. Type questions in the sidebar chat interface
2. Ask about your schedule, events, or general calendar queries
3. Get context-aware responses based on your calendar data

### Calendar Views
- Month View: Overview of all events
- Week View: Detailed weekly schedule
- Day View: Hour-by-hour daily view
- List View: Sequential list of events

## Architecture 🏗️

- Frontend: Streamlit
- Database: ChromaDB (vector database)
- AI Model: LLaMA via Ollama
- Calendar: streamlit-calendar component
- Vector Embeddings: LLaMA embeddings

## Dependencies 📦

- streamlit
- chromadb
- pandas
- streamlit-calendar
- requests
- Ollama with LLaMA model

## Features in Detail ⚙️

### Vector Search
- Events are stored with vector embeddings
- Semantic search enables natural language queries
- Similar events can be found based on context

### AI Chat
- Natural language understanding
- Context-aware responses
- Calendar-specific knowledge
- Persistent chat history

### Event Management
- Real-time updates
- Color coding
- Date validation
- Interactive interface

## Contributing 🤝

Feel free to:
1. Open issues for bugs or suggestions
2. Submit pull requests for improvements
3. Add new features or enhancements

## Notes 📝

- Ensure Ollama is running before starting the application
- Vector embeddings require sufficient memory
- First-time startup might be slower due to model loading

## License

MIT License - feel free to use and modify for your needs.
