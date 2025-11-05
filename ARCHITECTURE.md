# Architecture Documentation

## System Overview

The AI-Assisted WhatsApp Restaurant Bot is built using a modern agentic architecture that combines RAG (Retrieval-Augmented Generation) with stateful conversation management via LangGraph.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ Test UI (/test)│  │Admin Dashboard│  │  API Endpoints  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                  LangGraph Conversation Agent                │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              State Graph Workflow                     │   │
│  │                                                        │   │
│  │  Entry: classify_intent                               │   │
│  │         │                                              │   │
│  │         ├──→ handle_booking                           │   │
│  │         ├──→ handle_menu_inquiry (RAG)                │   │
│  │         ├──→ provide_info                             │   │
│  │         └──→ general_chat                             │   │
│  │                                                        │   │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                   │
│  ┌────────────────────────┴────────────────────────────┐    │
│  │          Conversation State (TypedDict)              │    │
│  │  - messages: List[BaseMessage]                       │    │
│  │  - phone_number: str                                 │    │
│  │  - current_intent: str                               │    │
│  │  - booking_data: dict                                │    │
│  │  - next_action: str                                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────┬────────────────┬─────────────────────┘
                       │                │
            ┌──────────┴────────┐   ┌──┴──────────────┐
            │   RAG System      │   │  LangChain Tools │
            └──────────┬────────┘   └──┬──────────────┘
                       │                │
            ┌──────────┴────────┐       │
            │  ChromaDB         │       ├─→ CreateBookingTool
            │  Vector Store     │       └─→ MenuQueryTool
            └───────────────────┘
```

## Core Components

### 1. State Management (LangGraph)

The system uses LangGraph's `StateGraph` for managing conversation flow:

**ConversationState Schema:**
```python
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    phone_number: str                         # User identifier
    current_intent: str                       # Classified intent
    booking_data: dict                        # Booking information being collected
    next_action: str                          # Next step in workflow
```

**State Persistence:**
- Uses `SqliteSaver` for checkpointing conversation state
- Maintains conversation history per `thread_id` (phone number)
- Enables multi-turn conversations with context retention

### 2. Intent Classification and Routing

**Intent Classification Node:**
```python
def classify_intent(state: ConversationState) -> ConversationState
```

Analyzes user messages and classifies into:
- `booking` - Table reservation requests
- `menu_inquiry` - Questions about food, prices, dietary options
- `info_inquiry` - Restaurant details (hours, location, etc.)
- `general_chat` - Greetings and casual conversation

**Routing Logic:**
Uses conditional edges to route to appropriate handlers:
```python
workflow.add_conditional_edges(
    "classify_intent",
    route_conversation,
    {
        "handle_booking": "handle_booking",
        "handle_menu_inquiry": "handle_menu_inquiry",
        "provide_info": "provide_info",
        "general_chat": "general_chat"
    }
)
```

### 3. RAG (Retrieval-Augmented Generation) System

**Architecture:**

```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store
                                                             │
                                                             ↓
User Query → Embedding → Similarity Search → Context → LLM Response
```

**Implementation Details:**

**a) Text Processing:**
```python
class MenuRAGSystem:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
```
- Splits menu text into 500-character chunks
- 50-character overlap ensures context continuity

**b) Embeddings:**
```python
embeddings = OllamaEmbeddings(
    model="mistral:latest",
    base_url="http://localhost:11434"
)
```
- Uses Ollama's Mistral model for generating embeddings
- Vector dimension: 4096 (Mistral default)

**c) Vector Storage:**
```python
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=self.embeddings,
    collection_name="restaurant_menu",
    persist_directory=VECTOR_DB_PATH
)
```
- ChromaDB persists vectors to disk
- Enables semantic search with cosine similarity

**d) Query Process:**
```python
def query_menu(self, query: str, k: int = 3) -> List[str]:
    docs = self.vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
```
- Returns top-k most relevant chunks
- Results passed to LLM for natural language generation

### 4. Conversation Flow Nodes

**a) Handle Menu Inquiry:**
```python
def handle_menu_inquiry(state: ConversationState) -> ConversationState
```
1. Extracts user query from message history
2. Calls `MenuQueryTool` for RAG retrieval
3. Constructs prompt with retrieved context
4. Generates natural response via LLM
5. Returns updated state with AI response

**b) Handle Booking:**
```python
def handle_booking(state: ConversationState) -> ConversationState
```
1. Extracts booking details from message (regex patterns)
2. Updates `booking_data` in state
3. Checks for missing required fields: `name`, `party_size`, `date`, `time`
4. If complete: creates booking via `CreateBookingTool`
5. If incomplete: asks for next missing field
6. Maintains state across multiple turns

**c) Provide Info:**
```python
def provide_info(state: ConversationState) -> ConversationState
```
- Loads restaurant configuration from JSON
- Returns formatted information (hours, location, contact)

**d) General Chat:**
```python
def general_chat(state: ConversationState) -> ConversationState
```
- Uses LLM for natural conversation
- Maintains restaurant persona (Maria)
- Guides users toward bookings or menu inquiries

### 5. Tools (LangChain)

**MenuQueryTool:**
```python
class MenuQueryTool(BaseTool):
    name = "query_menu"
    description = "Search restaurant menu using RAG"

    def _run(self, query: str) -> str:
        menu_results = rag_system.query_menu(query, k=3)
        return f"Based on our menu: {menu_info}"
```

**CreateBookingTool:**
```python
class CreateBookingTool(BaseTool):
    name = "create_booking"
    description = "Create restaurant reservation with validation"

    def _run(self, booking_data: str) -> str:
        # Validates party size against max_guests
        # Generates booking reference
        # Persists to bookings.json
```

### 6. Data Persistence

**FileDataManager:**
- `bookings.json`: All confirmed reservations
- `conversations.json`: Last 100 conversation exchanges
- `restaurant_config.json`: Restaurant settings
- `restaurant_bot.db`: SQLite for LangGraph checkpoints
- `chroma_db/`: ChromaDB vector database

### 7. LLM Integration

**Ollama Configuration:**
```python
llm = Ollama(
    model="mistral:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)
```
- Local inference (no API costs)
- 7B parameter model balances quality and speed
- Temperature 0.7 for natural but controlled responses

## Request Flow Example

### Menu Query Flow:

```
1. User: "Do you have vegetarian pasta?"
   ↓
2. FastAPI endpoint receives message
   ↓
3. Create ConversationState with HumanMessage
   ↓
4. LangGraph invokes workflow with state
   ↓
5. classify_intent: detects "menu_inquiry" (keyword: "vegetarian", "pasta")
   ↓
6. route_conversation: routes to handle_menu_inquiry
   ↓
7. handle_menu_inquiry:
   - Calls MenuQueryTool
   - RAG system queries vector DB with "vegetarian pasta"
   - ChromaDB returns top 3 relevant chunks
   - Constructs prompt with menu context
   - LLM generates natural response
   ↓
8. State updated with AIMessage response
   ↓
9. Response returned to user: "Yes, we offer Penne Primavera..."
```

### Booking Flow (Multi-turn):

```
Turn 1:
User: "I need a table"
→ classify_intent: booking
→ handle_booking: missing all fields
→ Response: "How many people will be dining with us?"

Turn 2:
User: "4 people"
→ Extract party_size=4
→ Still missing: name, date, time
→ Response: "What date would you prefer?"

Turn 3:
User: "Tomorrow at 8pm"
→ Extract date="tomorrow", time="8pm"
→ Still missing: name
→ Response: "Great! Just need a name for the reservation."

Turn 4:
User: "John Smith"
→ Extract name="John Smith"
→ All fields complete
→ CreateBookingTool invoked
→ Response: "Booking confirmed! Reference: BV001..."
```

## Scaling Considerations

### Current Limitations:
- In-memory conversation state (per server instance)
- File-based persistence (not production-ready)
- Single-threaded request handling
- Local LLM (slower than API-based models)

### Production Enhancements:
1. **Distributed State:** Redis for conversation state
2. **Database:** PostgreSQL instead of JSON files
3. **Queue System:** Celery for async processing
4. **Cloud LLM:** OpenAI/Anthropic for faster responses
5. **Monitoring:** Logging, tracing, metrics
6. **Caching:** Cache common menu queries

## Security Considerations

1. **Input Validation:** Sanitize user inputs for XSS/injection
2. **Rate Limiting:** Prevent abuse of endpoints
3. **Authentication:** Secure admin dashboard
4. **Data Privacy:** Encrypt sensitive booking information
5. **LLM Safety:** Add content filtering for inappropriate queries

## Testing Strategy

1. **Unit Tests:** Individual node functions
2. **Integration Tests:** Full conversation flows
3. **RAG Tests:** Vector retrieval accuracy
4. **Load Tests:** Concurrent conversation handling
5. **UI Tests:** Web interface interaction

## Future Enhancements

1. **Multi-language Support:** Detect language and respond accordingly
2. **Voice Integration:** WhatsApp voice message processing
3. **Image Recognition:** Menu item images
4. **Advanced Analytics:** Conversation insights and booking patterns
5. **A/B Testing:** Response quality optimization
