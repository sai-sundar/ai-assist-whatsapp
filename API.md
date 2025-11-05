# API Documentation

Complete API reference for the AI-Assisted WhatsApp Restaurant Bot.

## Base URL

```
http://localhost:8000
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/test` | GET | Interactive test UI for chat simulation |
| `/test-chat` | POST | Send test messages to the bot |
| `/upload-menu` | POST | Upload and process menu PDF |
| `/admin` | GET | Admin dashboard for monitoring |
| `/health` | GET | System health check |

---

## Endpoints

### 1. Test Interface

**GET** `/test`

Opens an interactive web interface for testing the restaurant bot.

**Response**: HTML page with chat interface

**Features**:
- Real-time chat simulation
- Menu upload interface
- Quick test message buttons
- Conversation history display

**Example Usage**:
```bash
# Open in browser
http://localhost:8000/test
```

**UI Components**:
- Chat container with message history
- Message input field
- File upload for menu PDFs
- Quick test buttons for common queries

---

### 2. Test Chat

**POST** `/test-chat`

Send a message to the bot and receive a response.

**Request Body**:
```json
{
  "message": "string",
  "phone_number": "string"
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User's message to the bot |
| phone_number | string | Yes | User identifier (conversation thread ID) |

**Response**:
```json
{
  "success": true,
  "message": "User's original message",
  "response": "Bot's response",
  "debug_info": {
    "intent": "booking | menu_inquiry | info_inquiry | general_chat",
    "booking_data": {},
    "menu_uploaded": true/false
  }
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the request was successful |
| message | string | Echo of user's message |
| response | string | Bot's generated response |
| debug_info | object | Additional debugging information |

**Example Request**:
```bash
curl -X POST http://localhost:8000/test-chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need a table for 4 people tomorrow at 8pm",
    "phone_number": "test-user-123"
  }'
```

**Example Response**:
```json
{
  "success": true,
  "message": "I need a table for 4 people tomorrow at 8pm",
  "response": "Great! Just need a name for the reservation.",
  "debug_info": {
    "intent": "booking",
    "booking_data": {
      "party_size": 4,
      "date": "tomorrow",
      "time": "8pm"
    },
    "menu_uploaded": true
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

**Common Intents and Example Messages**:

**Booking Intent**:
```json
{
  "message": "I want to book a table",
  "phone_number": "user123"
}
// Response: "How many people will be dining with us?"
```

**Menu Inquiry**:
```json
{
  "message": "Do you have vegetarian pasta?",
  "phone_number": "user123"
}
// Response: "Based on our menu, we offer..." (uses RAG)
```

**Info Inquiry**:
```json
{
  "message": "What are your hours?",
  "phone_number": "user123"
}
// Response: Restaurant information from config
```

**General Chat**:
```json
{
  "message": "Hello!",
  "phone_number": "user123"
}
// Response: Friendly greeting from Maria
```

---

### 3. Upload Menu

**POST** `/upload-menu`

Upload a menu PDF for RAG processing.

**Request**:
- Content-Type: `multipart/form-data`
- File field: `file`

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | PDF file containing menu (must be .pdf) |

**Response**:
```json
{
  "success": true,
  "message": "Menu 'filename.pdf' uploaded and processed successfully!"
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error description"
}
```

**Example Request (cURL)**:
```bash
curl -X POST http://localhost:8000/upload-menu \
  -F "file=@/path/to/menu.pdf"
```

**Example Request (JavaScript)**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/upload-menu', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Example Request (Python)**:
```python
import requests

with open('menu.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/upload-menu', files=files)
    print(response.json())
```

**Processing Steps**:
1. Validates file is a PDF
2. Saves uploaded file to `./uploaded_menus/`
3. Extracts text using PyPDF2
4. Splits text into chunks (500 chars, 50 overlap)
5. Generates embeddings using Ollama Mistral
6. Stores in ChromaDB vector database
7. Updates `restaurant_config.json` with `menu_uploaded: true`

**Common Errors**:
| Error | Cause | Solution |
|-------|-------|----------|
| "Please upload a PDF file" | File extension is not .pdf | Use a PDF file |
| "No text extracted from PDF" | PDF is image-based or protected | Use OCR or text-based PDF |
| "Embeddings not initialized" | Ollama not running | Start Ollama service |
| "Vector store creation error" | ChromaDB issue | Check disk space and permissions |

---

### 4. Admin Dashboard

**GET** `/admin`

Administrative dashboard for monitoring system status.

**Response**: HTML page with system statistics

**Dashboard Sections**:

1. **Statistics Cards**:
   - Total bookings count
   - Recent conversations count
   - RAG system status
   - Test mode indicator

2. **RAG System Status**:
   - Menu upload status (Yes/No)
   - Vector database status
   - Links to test interface and upload

3. **Restaurant Configuration**:
   - Restaurant name
   - Cuisine type
   - Operating hours
   - Max guests capacity
   - Location and contact

4. **Recent Bookings Table**:
   - Booking reference
   - Customer name
   - Party size
   - Date and time
   - Phone number (last 4 digits)
   - Status

5. **Recent Conversations**:
   - Last 20 conversations
   - User messages and bot responses
   - Timestamps
   - Phone numbers (masked)

**Example Usage**:
```bash
# Open in browser
http://localhost:8000/admin
```

---

### 5. Health Check

**GET** `/health`

System health and status endpoint.

**Response**:
```json
{
  "status": "OK",
  "version": "3.0-RAG-Test",
  "timestamp": "2025-01-15T10:30:45.123456",
  "menu_uploaded": true,
  "rag_system": "active"
}
```

**Response Fields**:
| Field | Type | Description |
|-------|------|-------------|
| status | string | Always "OK" if server is running |
| version | string | Application version |
| timestamp | string | ISO 8601 timestamp |
| menu_uploaded | boolean | Whether a menu PDF has been uploaded |
| rag_system | string | RAG system status |

**Example Request**:
```bash
curl http://localhost:8000/health
```

**Use Cases**:
- Monitoring and uptime checks
- CI/CD health validation
- Load balancer health probes

---

## Data Models

### Conversation State

Internal state maintained by LangGraph:

```python
{
  "messages": [
    {
      "type": "human",
      "content": "User message"
    },
    {
      "type": "ai",
      "content": "Bot response"
    }
  ],
  "phone_number": "user-identifier",
  "current_intent": "booking | menu_inquiry | info_inquiry | general_chat",
  "booking_data": {
    "name": "string",
    "party_size": 4,
    "date": "string",
    "time": "string",
    "phone_number": "string"
  },
  "next_action": "string"
}
```

### Booking Record

Stored in `bookings.json`:

```json
{
  "booking_reference": "BV001",
  "name": "John Smith",
  "party_size": 4,
  "date": "tomorrow",
  "time": "8pm",
  "phone_number": "+1234567890",
  "timestamp": "2025-01-15T10:30:45.123456",
  "status": "confirmed"
}
```

### Conversation Record

Stored in `conversations.json`:

```json
{
  "phone_number": "user123",
  "message": "Do you have vegetarian options?",
  "response": "Yes, we offer several vegetarian dishes...",
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

### Restaurant Configuration

Stored in `restaurant_config.json`:

```json
{
  "name": "Bella Vista Restaurant",
  "cuisine_type": "Italian with Luxembourg touches",
  "hours": "Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM",
  "location": "15 Rue de la Paix, Luxembourg City",
  "phone": "+352 12 34 56 78",
  "max_guests": 20,
  "policies": "Cancellation required 2 hours in advance",
  "menu_uploaded": false
}
```

---

## Intent Classification

The system automatically classifies user messages into intents:

### Booking Intent
**Triggers**: "book", "reserve", "table", "reservation"

**Example Messages**:
- "I need a table"
- "Can I make a reservation?"
- "Book a table for 4"

### Menu Inquiry
**Triggers**: "menu", "food", "dish", "price", "vegan", "vegetarian", "cost", "allergen", "spicy"

**Example Messages**:
- "Do you have vegetarian pasta?"
- "What's on the menu?"
- "How much is the steak?"
- "Any gluten-free options?"

### Info Inquiry
**Triggers**: "hours", "open", "close", "location", "address"

**Example Messages**:
- "What are your hours?"
- "Where are you located?"
- "When do you close?"

### General Chat
**Default**: Any message not matching above patterns

**Example Messages**:
- "Hello"
- "Thank you"
- "Sounds good"

---

## Booking Extraction Patterns

The system uses regex to extract booking details:

### Party Size Patterns
- "party of 4"
- "4 people"
- "table for 4"
- "4 guests"
- Just "4" (if context suggests booking)

### Date Patterns
- "today", "tomorrow", "tonight"
- "Monday", "Tuesday", etc.
- "January 15th", "March 3"

### Time Patterns
- "8:30pm", "8pm"
- "20:00", "8:30 AM"

### Name Extraction
- Short text (1-4 words) with no booking keywords or numbers
- Excludes common phrases: "yes", "no", "ok", "thanks"

---

## Rate Limiting

**Note**: Current version has no rate limiting. For production:

```python
# Recommended: Use slowapi or similar
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/test-chat")
@limiter.limit("10/minute")
async def test_chat(request: Request):
    # ...
```

---

## Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Endpoint not found |
| 500 | Internal Server Error |

**Error Response Format**:
```json
{
  "success": false,
  "error": "Description of error"
}
```

---

## CORS Configuration

**Note**: Current version has no CORS headers. For production with web clients:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## WebSocket Support

**Note**: Current version uses HTTP only. For real-time chat, consider WebSocket endpoints:

```python
from fastapi import WebSocket

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()
    # Handle real-time messaging
```

---

## Testing API Endpoints

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Send chat message
curl -X POST http://localhost:8000/test-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "phone_number": "test123"}'

# Upload menu
curl -X POST http://localhost:8000/upload-menu \
  -F "file=@menu.pdf"
```

### Using Python

```python
import requests

# Chat message
response = requests.post(
    'http://localhost:8000/test-chat',
    json={
        'message': 'I need a table for 4',
        'phone_number': 'test-user'
    }
)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
// Chat message
fetch('http://localhost:8000/test-chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What are your hours?',
    phone_number: 'test-user'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Future API Enhancements

Planned endpoints for future versions:

- `POST /api/booking` - Create booking (structured API)
- `GET /api/booking/{id}` - Get booking details
- `DELETE /api/booking/{id}` - Cancel booking
- `POST /api/auth/login` - Admin authentication
- `GET /api/analytics` - Conversation analytics
- `POST /api/webhook/whatsapp` - WhatsApp Business API webhook

---

## Support

For API-related questions or issues, please open an issue on GitHub with:
- Endpoint being called
- Request payload
- Response received
- Expected behavior
