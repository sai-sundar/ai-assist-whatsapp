# AI-Assisted WhatsApp Restaurant Bot

A sophisticated AI-powered restaurant booking assistant with RAG (Retrieval-Augmented Generation) capabilities for intelligent menu queries. Built with LangChain, LangGraph, and FastAPI.

## Overview

This project implements an intelligent restaurant assistant bot that can:
- Handle restaurant table bookings with natural language processing
- Answer menu-related questions using RAG (vector database for menu items)
- Provide restaurant information (hours, location, contact)
- Maintain conversation context and state management
- Process PDF menu uploads and create searchable embeddings

## Key Features

### 1. RAG-Enhanced Menu Intelligence
- Upload restaurant menus in PDF format
- Automatic text extraction and chunking
- Vector embeddings using Ollama (Mistral model)
- Semantic search for menu items, prices, and dietary information
- ChromaDB vector storage for efficient retrieval

### 2. Agentic Conversation Flow (LangGraph)
- State-based conversation management
- Intent classification (booking, menu inquiry, general chat, info)
- Context-aware routing and response generation
- Persistent conversation memory with SQLite checkpointing

### 3. Booking Management
- Natural language date/time extraction
- Party size validation
- Booking confirmation with reference numbers
- JSON-based persistence (bookings.json)

### 4. Testing Interface
- Interactive web UI for testing conversations
- Real-time chat simulation
- Menu upload interface
- Admin dashboard for monitoring bookings and conversations

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
    ┌────┴─────┐
    │ LangGraph │
    │  Agent    │
    └────┬─────┘
         │
    ┌────┴────────────────┐
    │                     │
┌───┴───┐          ┌─────┴─────┐
│  RAG  │          │   Tools   │
│System │          │ (Booking, │
│       │          │  Menu)    │
└───┬───┘          └───────────┘
    │
┌───┴────────┐
│  ChromaDB  │
│  Vectors   │
└────────────┘
```

### Components

- **FastAPI**: Web framework for API endpoints and UI
- **LangGraph**: Agentic workflow orchestration with state graphs
- **LangChain**: LLM integration and tool management
- **Ollama**: Local LLM inference (Mistral model)
- **ChromaDB**: Vector database for menu embeddings
- **PyPDF2**: PDF text extraction for menu processing

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Mistral model pulled: `ollama pull mistral:7b`
- Ollama service running on `http://localhost:11434`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sai-sundar/ai-assist-whatsapp.git
   cd ai-assist-whatsapp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional RAG dependencies**
   ```bash
   pip install langchain langchain-community langgraph chromadb pypdf2 ollama
   ```

5. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

6. **Pull the Mistral model**
   ```bash
   ollama pull mistral:7b
   ```

## Usage

### Start the Application

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Available Endpoints

- **Test Interface**: http://localhost:8000/test
  - Interactive chat UI for testing the bot
  - Upload menu PDFs
  - Real-time conversation testing

- **Admin Dashboard**: http://localhost:8000/admin
  - View all bookings
  - Monitor conversations
  - Check RAG system status
  - Restaurant configuration

- **API Endpoints**:
  - `POST /test-chat` - Send test messages
  - `POST /upload-menu` - Upload menu PDF
  - `GET /health` - System health check

### Testing the Bot

1. Navigate to http://localhost:8000/test
2. Upload a menu PDF (optional but recommended for menu queries)
3. Try example queries:
   - "Do you have vegetarian pasta?"
   - "I need a table for 4 people tomorrow at 8pm"
   - "What are your hours?"
   - "What desserts do you have?"

### Example Conversations

**Menu Query:**
```
User: Do you have any vegan options?
Bot: Based on our menu, we offer several vegan dishes including...
```

**Booking Flow:**
```
User: I'd like to book a table for 4 people
Bot: How many people will be dining with us?
User: 4 people
Bot: What date would you prefer?
User: Tomorrow at 8pm
Bot: Great! Just need a name for the reservation.
User: John Smith
Bot: Perfect! Booking confirmed! Reference: BV001...
```

## Project Structure

```
ai-assist-whatsapp/
├── main.py                    # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── chroma_db/                 # Vector database storage
├── uploaded_menus/            # Uploaded menu PDFs
├── bookings.json              # Booking records
├── conversations.json         # Conversation history
├── restaurant_config.json     # Restaurant configuration
└── restaurant_bot.db          # SQLite checkpoint storage
```

## Configuration

Edit `restaurant_config.json` to customize:

```json
{
  "name": "Your Restaurant Name",
  "cuisine_type": "Cuisine Description",
  "hours": "Opening hours",
  "location": "Address",
  "phone": "Contact number",
  "max_guests": 20,
  "policies": "Cancellation and other policies"
}
```

## Technologies Used

- **FastAPI** - Modern web framework
- **LangChain** - LLM orchestration framework
- **LangGraph** - Agentic workflow graphs
- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database
- **PyPDF2** - PDF processing
- **SQLite** - Conversation state persistence

## Development Roadmap

- [ ] WhatsApp Business API integration
- [ ] Multi-language support
- [ ] Advanced booking conflict detection
- [ ] Email/SMS confirmations
- [ ] Analytics dashboard
- [ ] Menu item image recognition

## Troubleshooting

**RAG System Errors:**
- Ensure Ollama is running: `curl http://localhost:11434`
- Check if Mistral model is available: `ollama list`
- Verify PDF text extraction (some PDFs may be image-based)

**Booking Not Saving:**
- Check file permissions for bookings.json
- Ensure the directory is writable

**LLM Timeout:**
- Increase Ollama timeout settings
- Check system resources (RAM, CPU)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is available for educational and testing purposes.

## Support

For issues or questions, please open an issue on GitHub.

## Acknowledgments

- Built with LangChain and LangGraph
- Powered by Ollama and Mistral LLM
- Vector search by ChromaDB
