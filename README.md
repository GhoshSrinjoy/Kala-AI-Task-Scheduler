# Smart Calendar with AI Assistant 🗓️

A modern, AI-powered calendar application built with Streamlit that combines traditional calendar functionality with semantic search and natural language interactions.

## Features 🌟

### Calendar Management
- Multiple calendar views (Month, Week, Day, List)
- Easy event creation and deletion
- Color-coded events
- Drag-and-drop interface
- Interactive event management panel

### AI Integration
- Semantic search through events using vector embeddings
- Natural language chat interface for calendar queries
- Context-aware responses using LLaMA language model
- Real-time event recommendations and insights

### Technical Features
- Vector database (ChromaDB) for semantic storage
- LLaMA integration for natural language understanding
- Real-time updates and caching
- Responsive UI with Streamlit
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
