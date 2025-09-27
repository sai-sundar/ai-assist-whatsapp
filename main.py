# RAG-Enabled Test System for Restaurant Bot
# File: main.py

from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import Response, HTMLResponse
from datetime import datetime
from typing import List, Dict, Any, Annotated, Literal, Optional
import uvicorn
import re
import json
import os
import PyPDF2
from io import BytesIO

# LangChain imports
from langchain_community.llms import Ollama
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

# Initialize FastAPI app
app = FastAPI(title="RAG Restaurant Bot Test System", version="3.0.0")

# File paths
BOOKINGS_FILE = "bookings.json"
CONVERSATIONS_FILE = "conversations.json"
RESTAURANT_CONFIG_FILE = "restaurant_config.json"
VECTOR_DB_PATH = "./chroma_db"
UPLOADED_MENUS_DIR = "./uploaded_menus"

def init_system():
    """Initialize the test system"""
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    os.makedirs(UPLOADED_MENUS_DIR, exist_ok=True)
    
    # Default restaurant config
    default_config = {
        "name": "Bella Vista Restaurant",
        "cuisine_type": "Italian with Luxembourg touches",
        "hours": "Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM, Closed Sundays",
        "location": "15 Rue de la Paix, Luxembourg City",
        "phone": "+352 12 34 56 78",
        "max_guests": 20,
        "policies": "Cancellation required 2 hours in advance",
        "menu_uploaded": False
    }
    
    if not os.path.exists(RESTAURANT_CONFIG_FILE):
        with open(RESTAURANT_CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    if not os.path.exists(BOOKINGS_FILE):
        with open(BOOKINGS_FILE, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump([], f)

# RAG System
class MenuRAGSystem:
    """RAG system for menu information"""
    
    def __init__(self):
        try:
            self.embeddings = OllamaEmbeddings(
                model="mistral:latest",
                base_url="http://localhost:11434"
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            self.vectorstore = None
            print("RAG system initialized successfully")
        except Exception as e:
            print(f"RAG system initialization error: {e}")
            self.embeddings = None
            
    def load_menu_from_pdf(self, pdf_file: bytes) -> dict:
        """Extract menu text from PDF and store in vector database"""
        result = {"success": False, "error": "", "details": ""}
        
        try:
            if self.embeddings is None:
                result["error"] = "Embeddings not initialized - check Ollama connection"
                return result
            
            # Try different PDF processing approaches
            menu_text = ""
            
            # Method 1: PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    menu_text += page_text + "\n"
                    print(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                    
                if len(menu_text.strip()) < 100:
                    result["error"] = "PDF appears to be mostly images or protected text"
                    result["details"] = f"Only extracted {len(menu_text)} characters"
                    return result
                    
            except Exception as pdf_error:
                result["error"] = f"PDF reading error: {str(pdf_error)}"
                return result
            
            print(f"Total extracted text: {len(menu_text)} characters")
            
            # Clean and validate extracted text
            menu_text = menu_text.strip()
            if not menu_text:
                result["error"] = "No text extracted from PDF"
                return result
            
            # Split text into chunks
            try:
                docs = self.text_splitter.split_text(menu_text)
                print(f"Created {len(docs)} document chunks")
                
                if len(docs) == 0:
                    result["error"] = "No document chunks created"
                    return result
                
            except Exception as split_error:
                result["error"] = f"Text splitting error: {str(split_error)}"
                return result
            
            # Create documents with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": "menu", "chunk_id": i}
                )
                for i, chunk in enumerate(docs)
            ]
            
            # Create vector store
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name="restaurant_menu",
                    persist_directory=VECTOR_DB_PATH
                )
                
                print(f"Vector store created with {len(documents)} chunks")
                
            except Exception as vector_error:
                result["error"] = f"Vector store creation error: {str(vector_error)}"
                return result
            
            # Update config to mark menu as uploaded
            try:
                with open(RESTAURANT_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                config['menu_uploaded'] = True
                with open(RESTAURANT_CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as config_error:
                print(f"Config update error: {config_error}")
            
            result["success"] = True
            result["details"] = f"Successfully processed {len(documents)} chunks from {len(menu_text)} characters"
            print(f"Menu loaded successfully: {len(documents)} chunks")
            return result
            
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            print(f"Menu loading error: {e}")
            return result
    
    def query_menu(self, query: str, k: int = 3) -> List[str]:
        """Query menu information"""
        try:
            if not self.vectorstore:
                # Try to load existing vectorstore
                if os.path.exists(VECTOR_DB_PATH) and self.embeddings:
                    try:
                        self.vectorstore = Chroma(
                            collection_name="restaurant_menu",
                            embedding_function=self.embeddings,
                            persist_directory=VECTOR_DB_PATH
                        )
                        print("Loaded existing vector store")
                    except Exception as e:
                        print(f"Error loading existing vector store: {e}")
                        return ["Menu not available - please upload a menu first"]
            
            if not self.vectorstore:
                return ["Menu not loaded. Please upload a menu first."]
            
            # Search for relevant menu items
            docs = self.vectorstore.similarity_search(query, k=k)
            results = [doc.page_content for doc in docs]
            
            print(f"RAG Query: '{query}' returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result[:100]}...")
            
            return results
            
        except Exception as e:
            print(f"Error querying menu: {e}")
            return [f"Error searching menu: {str(e)}"]

# Initialize RAG system
rag_system = MenuRAGSystem()

# Data managers
class FileDataManager:
    @staticmethod
    def load_restaurant_config() -> dict:
        try:
            with open(RESTAURANT_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    @staticmethod
    def save_conversation(phone_number: str, message: str, response: str):
        try:
            with open(CONVERSATIONS_FILE, 'r') as f:
                conversations = json.load(f)
            
            conversations.append({
                "phone_number": phone_number,
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            conversations = conversations[-100:]  # Keep last 100
            
            with open(CONVERSATIONS_FILE, 'w') as f:
                json.dump(conversations, f, indent=2)
                
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    @staticmethod
    def save_booking(booking_data: dict) -> str:
        try:
            with open(BOOKINGS_FILE, 'r') as f:
                bookings = json.load(f)
            
            booking_id = len(bookings) + 1
            booking_ref = f"BV{booking_id:03d}"
            
            booking_entry = {
                **booking_data,
                "booking_reference": booking_ref,
                "timestamp": datetime.now().isoformat(),
                "status": "confirmed"
            }
            
            bookings.append(booking_entry)
            
            with open(BOOKINGS_FILE, 'w') as f:
                json.dump(bookings, f, indent=2)
            
            print(f"✅ Booking saved: {booking_ref}")
            return booking_ref
            
        except Exception as e:
            print(f"Error saving booking: {e}")
            return "BV999"
    
    @staticmethod
    def get_all_bookings() -> List[dict]:
        try:
            with open(BOOKINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    
    @staticmethod
    def get_all_conversations() -> List[dict]:
        try:
            with open(CONVERSATIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []

# Conversation state
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    phone_number: str
    current_intent: str
    booking_data: dict
    next_action: str

# Enhanced tools with RAG
class MenuQueryTool(BaseTool):
    name = "query_menu"
    description = "Search the restaurant menu for specific dishes, prices, or dietary information using RAG"
    
    def _run(self, query: str) -> str:
        try:
            # Query the menu using RAG
            menu_results = rag_system.query_menu(query, k=3)
            
            if menu_results and menu_results[0] != "Menu not loaded. Please upload a menu first.":
                # Combine results for LLM processing
                menu_info = "\n".join(menu_results[:3])
                return f"Based on our menu: {menu_info}"
            else:
                return "Menu information not available. Please contact the restaurant for details."
                
        except Exception as e:
            print(f"Menu query error: {e}")
            return "Unable to search menu at the moment."

class CreateBookingTool(BaseTool):
    name = "create_booking"
    description = "Create a restaurant reservation with validation"
    
    def _run(self, booking_data: str) -> str:
        try:
            data = json.loads(booking_data)
            
            # Validate party size
            restaurant_config = FileDataManager.load_restaurant_config()
            max_guests = restaurant_config.get("max_guests", 20)
            
            if data.get("party_size", 0) > max_guests:
                return f"Sorry, we can accommodate a maximum of {max_guests} guests per reservation."
            
            # Save booking
            booking_ref = FileDataManager.save_booking(data)
            
            return f"Booking confirmed! Reference: {booking_ref}. Table for {data.get('party_size')} people on {data.get('date')} at {data.get('time')} under {data.get('name')}."
            
        except Exception as e:
            print(f"Booking tool error: {e}")
            return "Sorry, I had trouble creating the booking. Please try again."

# Initialize tools and LLM
tools = [CreateBookingTool(), MenuQueryTool()]
data_manager = FileDataManager()

llm = Ollama(
    model="mistral:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Graph functions (same as before but with enhanced menu handling)
def classify_intent(state: ConversationState) -> ConversationState:
    if not state["messages"]:
        return state
    
    last_message = state["messages"][-1].content.lower()
    booking_data = state.get("booking_data", {})
    
    # Check for ongoing booking first
    if booking_data:
        required_fields = ["name", "party_size", "date", "time"]
        missing = [f for f in required_fields if f not in booking_data or not booking_data[f]]
        if missing:
            intent = "booking"
        else:
            intent = "general_chat"
    # Check for menu queries
    elif any(word in last_message for word in ["menu", "food", "dish", "price", "vegan", "vegetarian", "cost", "allergen", "spicy"]):
        intent = "menu_inquiry"
    # Check for booking intent
    elif any(word in last_message for word in ["book", "reserve", "table", "reservation"]):
        intent = "booking"
    # Check for restaurant info
    elif any(word in last_message for word in ["hours", "open", "close", "location", "address"]):
        intent = "info_inquiry"
    else:
        intent = "general_chat"
    
    print(f"🔍 Intent classified: {intent}")
    
    return {
        **state,
        "current_intent": intent,
        "next_action": "route_conversation"
    }

def route_conversation(state: ConversationState) -> Literal["handle_booking", "handle_menu_inquiry", "provide_info", "general_chat"]:
    intent = state["current_intent"]
    
    if intent == "booking":
        return "handle_booking"
    elif intent == "menu_inquiry":
        return "handle_menu_inquiry"
    elif intent == "info_inquiry":
        return "provide_info"
    else:
        return "general_chat"

def handle_menu_inquiry(state: ConversationState) -> ConversationState:
    """Handle menu-related queries using RAG"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Query menu using RAG tool
    menu_tool = MenuQueryTool()
    menu_info = menu_tool._run(last_message)
    
    # Generate natural response using LLM with menu context
    prompt = f"""You are Maria, a friendly restaurant assistant. A customer is asking about the menu.

Customer query: {last_message}

Menu information found: {menu_info}

Provide a helpful, natural response about the menu items. If specific items are found, mention them with prices when available. Keep it conversational and offer to help with reservations. Be warm and friendly."""
    
    try:
        response = llm.invoke(prompt)
    except:
        response = f"{menu_info[:300]}... Would you like to make a reservation?"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "end"
    }

# Keep other functions similar to simplified version
def handle_booking(state: ConversationState) -> ConversationState:
    messages = state["messages"]
    booking_data = state.get("booking_data", {})
    
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not human_messages:
        return state
        
    last_human_message = human_messages[-1].content
    
    # Extract booking details
    extracted = extract_booking_details(last_human_message)
    booking_data.update(extracted)
    
    print(f"📋 Current booking data: {booking_data}")
    
    # Check completion
    required_fields = ["name", "party_size", "date", "time"]
    missing = [f for f in required_fields if f not in booking_data or not booking_data[f]]
    
    if not missing:
        booking_json = json.dumps({
            **booking_data,
            "phone_number": state["phone_number"]
        })
        
        result = tools[0]._run(booking_json)
        response = f"Perfect! {result}"
        next_action = "end"
    else:
        next_field = missing[0]
        if next_field == "party_size":
            response = "How many people will be dining with us?"
        elif next_field == "date":
            response = "What date would you prefer?"
        elif next_field == "time":
            response = "What time works best for you?"
        elif next_field == "name":
            response = "Great! Just need a name for the reservation."
        else:
            response = "Let me help you with that booking."
        
        next_action = "end"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "booking_data": booking_data,
        "next_action": next_action
    }

def provide_info(state: ConversationState) -> ConversationState:
    restaurant_config = FileDataManager.load_restaurant_config()
    
    response = f"""📍 {restaurant_config.get('name', 'Bella Vista Restaurant')}
📅 Hours: {restaurant_config.get('hours', 'Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM')}
🏠 Location: {restaurant_config.get('location', '15 Rue de la Paix, Luxembourg City')}
📞 Phone: {restaurant_config.get('phone', '+352 12 34 56 78')}

Would you like to make a reservation?"""
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "end"
    }

def general_chat(state: ConversationState) -> ConversationState:
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    restaurant_config = FileDataManager.load_restaurant_config()
    
    prompt = f"""You are Maria, a friendly staff member at {restaurant_config.get('name', 'Bella Vista Restaurant')}.

Restaurant info: {restaurant_config.get('cuisine_type', 'Italian with Luxembourg touches')}
Hours: {restaurant_config.get('hours', 'Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM')}

Customer says: {last_message}

Respond warmly and naturally (1-2 sentences). Guide towards menu questions or reservations when appropriate."""
    
    try:
        response = llm.invoke(prompt)
    except:
        response = "Hi! I'm here to help with questions about our restaurant. Would you like to see our menu or make a reservation?"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "end"
    }

def extract_booking_details(text: str) -> dict:
    """Extract booking information from text"""
    details = {}
    text_lower = text.lower()
    
    # Party size patterns
    party_patterns = [
        r'party\s+of\s+(\d{1,2})',
        r'(?:for\s+)?(\d{1,2})\s*(?:people|person|pax|guests?)',
        r'table\s+for\s+(\d{1,2})',
        r'^\s*(\d{1,2})\s*$'
    ]
    
    for pattern in party_patterns:
        match = re.search(pattern, text_lower)
        if match:
            details["party_size"] = int(match.group(1))
            break
    
    # Date patterns
    date_patterns = [
        r'\b(today|tomorrow|tonight)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:
                details["date"] = f"{match.group(1)} {match.group(2)}"
            else:
                details["date"] = match.group()
            break
    
    # Time patterns
    time_patterns = [
        r'\b(\d{1,2}):(\d{2})\s*(?:am|pm)\b',
        r'\b(\d{1,2})\s*(?:am|pm)\b'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:
                details["time"] = f"{match.group(1)}:{match.group(2)}"
                ampm_match = re.search(r'\b(?:am|pm)\b', text_lower)
                if ampm_match:
                    details["time"] += f" {ampm_match.group()}"
            else:
                details["time"] = match.group()
            break
    
    # Name extraction
    words = text.strip().split()
    booking_keywords = ['table', 'book', 'reserve', 'party', 'people', 'reservation']
    
    if (len(words) <= 4 and 
        not any(keyword in text_lower for keyword in booking_keywords) and
        not any(char.isdigit() for char in text) and
        len(text.strip()) > 1 and
        not any(details.values())):
        
        common_phrases = ['yes', 'no', 'ok', 'okay', 'thanks', 'hello', 'hi']
        if text_lower not in common_phrases:
            details["name"] = text.strip()
    
    return details

# Build conversation graph
def build_conversation_graph():
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_booking", handle_booking)
    workflow.add_node("handle_menu_inquiry", handle_menu_inquiry)
    workflow.add_node("provide_info", provide_info)
    workflow.add_node("general_chat", general_chat)
    
    workflow.set_entry_point("classify_intent")
    
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
    
    workflow.add_edge("handle_booking", END)
    workflow.add_edge("handle_menu_inquiry", END)
    workflow.add_edge("provide_info", END)
    workflow.add_edge("general_chat", END)
    
    memory = SqliteSaver.from_conn_string(":memory:")
    return workflow.compile(checkpointer=memory)

conversation_graph = build_conversation_graph()

@app.on_event("startup")
async def startup_event():
    init_system()
    print("🚀 RAG Test System started!")
    print("🧪 Test interface: http://localhost:8000/test")
    print("📤 Upload menu: http://localhost:8000/upload-menu")

@app.post("/upload-menu")
async def upload_menu(file: UploadFile = File(...)):
    """Upload and process menu PDF"""
    try:
        if not file.filename.endswith('.pdf'):
            return {"success": False, "error": "Please upload a PDF file"}
        
        # Read PDF content
        pdf_content = await file.read()
        
        # Save uploaded file
        file_path = os.path.join(UPLOADED_MENUS_DIR, file.filename)
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        
        # Process with RAG system
        if rag_system.load_menu_from_pdf(pdf_content):
            return {"success": True, "message": f"Menu '{file.filename}' uploaded and processed successfully!"}
        else:
            return {"success": False, "error": "Failed to process PDF"}
            
    except Exception as e:
        return {"success": False, "error": f"Upload error: {str(e)}"}

@app.get("/test", response_class=HTMLResponse)
async def test_interface():
    """Interactive test interface"""
    restaurant_config = data_manager.load_restaurant_config()
    menu_status = "✅ Uploaded" if restaurant_config.get('menu_uploaded', False) else "❌ Not uploaded"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Restaurant Bot Test Interface</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .chat-container {{ border: 1px solid #ddd; height: 400px; overflow-y: scroll; padding: 20px; margin: 20px 0; background: #f9f9f9; }}
            .message {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
            .user {{ background: #007bff; color: white; text-align: right; }}
            .bot {{ background: #28a745; color: white; }}
            .system {{ background: #ffc107; color: black; font-style: italic; }}
            input[type="text"] {{ width: 70%; padding: 10px; }}
            button {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            .upload-area {{ background: #e9ecef; padding: 20px; margin: 20px 0; border-radius: 5px; text-align: center; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background: #d4edda; color: #155724; }}
            .error {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <h1>RAG Restaurant Bot Test Interface</h1>
        
        <div class="status success">
            <strong>Restaurant:</strong> {restaurant_config.get('name', 'Not configured')}<br>
            <strong>Menu Status:</strong> {menu_status}
        </div>
        
        <div class="upload-area">
            <h3>Upload Menu (PDF)</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="menuFile" accept=".pdf" required>
                <button type="submit">Upload Menu</button>
            </form>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message system">Welcome! Test the RAG-enhanced restaurant bot. Try asking about menu items, making reservations, or general questions.</div>
        </div>
        
        <div>
            <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
            <button onclick="clearChat()">Clear Chat</button>
        </div>
        
        <div>
            <h3>Quick Test Messages:</h3>
            <button onclick="sendQuickMessage('Do you have vegetarian pasta?')">Menu Query</button>
            <button onclick="sendQuickMessage('I need a table for 4 people tomorrow at 8pm')">Booking Request</button>
            <button onclick="sendQuickMessage('What are your hours?')">Restaurant Info</button>
            <button onclick="sendQuickMessage('What desserts do you have?')">Dessert Query</button>
        </div>
        
        <script>
            let conversationId = 'test-' + Date.now();
            
            async function sendMessage() {{
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, 'user');
                input.value = '';
                
                try {{
                    const response = await fetch('/test-chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ message, phone_number: conversationId }})
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        addMessage(result.response, 'bot');
                        if (result.debug_info) {{
                            addMessage('Debug: ' + JSON.stringify(result.debug_info), 'system');
                        }}
                    }} else {{
                        addMessage('Error: ' + result.error, 'error');
                    }}
                }} catch (error) {{
                    addMessage('Connection error: ' + error.message, 'error');
                }}
            }}
            
            function sendQuickMessage(message) {{
                document.getElementById('messageInput').value = message;
                sendMessage();
            }}
            
            function addMessage(text, type) {{
                const container = document.getElementById('chatContainer');
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.textContent = text;
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }}
            
            function clearChat() {{
                const container = document.getElementById('chatContainer');
                container.innerHTML = '<div class="message system">Chat cleared. Start a new conversation.</div>';
                conversationId = 'test-' + Date.now();
            }}
            
            // Upload form handler
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('menuFile');
                formData.append('file', fileInput.files[0]);
                
                try {{
                    const response = await fetch('/upload-menu', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        alert('Menu uploaded successfully!');
                        location.reload();
                    }} else {{
                        alert('Upload failed: ' + result.error);
                    }}
                }} catch (error) {{
                    alert('Upload error: ' + error.message);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.post("/test-chat")
async def test_chat(request: Request):
    """Test chat endpoint for the web interface"""
    try:
        data = await request.json()
        message = data.get("message", "")
        phone_number = data.get("phone_number", "test-user")
        
        # Create initial state
        initial_state = ConversationState(
            messages=[HumanMessage(content=message)],
            phone_number=phone_number,
            current_intent="",
            booking_data={},
            next_action=""
        )
        
        # Run through conversation graph
        config = {
            "configurable": {"thread_id": phone_number},
            "recursion_limit": 50
        }
        result = conversation_graph.invoke(initial_state, config)
        
        # Get AI response
        ai_response = result["messages"][-1].content
        
        # Save conversation
        data_manager.save_conversation(phone_number, message, ai_response)
        
        return {
            "success": True,
            "message": message,
            "response": ai_response,
            "debug_info": {
                "intent": result.get("current_intent", ""),
                "booking_data": result.get("booking_data", {}),
                "menu_uploaded": data_manager.load_restaurant_config().get('menu_uploaded', False)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard with RAG system info"""
    bookings = data_manager.get_all_bookings()
    conversations = data_manager.get_all_conversations()[-20:]
    restaurant_config = data_manager.load_restaurant_config()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Restaurant Bot Admin</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #f8f9fa; }}
            .conversation {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .user {{ background: #007bff; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            .bot {{ background: #28a745; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            .config {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .rag-status {{ background: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #bee5eb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>RAG-Enhanced Restaurant Bot Admin</h1>
                <p>Test system with menu intelligence</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(bookings)}</div>
                    <div>Total Bookings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(conversations)}</div>
                    <div>Recent Conversations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">RAG</div>
                    <div>Menu Intelligence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">TEST</div>
                    <div>Mode</div>
                </div>
            </div>
            
            <div class="rag-status">
                <h3>RAG System Status</h3>
                <p><strong>Menu Uploaded:</strong> {'Yes' if restaurant_config.get('menu_uploaded', False) else 'No'}</p>
                <p><strong>Vector Database:</strong> {'Active' if os.path.exists(VECTOR_DB_PATH) else 'Not initialized'}</p>
                <p><strong>Test Interface:</strong> <a href="/test">http://localhost:8000/test</a></p>
                <p><strong>Menu Upload:</strong> <a href="/upload-menu">Upload new menu PDF</a></p>
            </div>
            
            <h2>Restaurant Configuration</h2>
            <div class="config">
                <p><strong>Name:</strong> {restaurant_config.get('name', 'N/A')}</p>
                <p><strong>Cuisine:</strong> {restaurant_config.get('cuisine_type', 'N/A')}</p>
                <p><strong>Hours:</strong> {restaurant_config.get('hours', 'N/A')}</p>
                <p><strong>Max Guests:</strong> {restaurant_config.get('max_guests', 'N/A')}</p>
                <p><strong>Location:</strong> {restaurant_config.get('location', 'N/A')}</p>
            </div>
            
            <h2>Recent Bookings (saved to bookings.json)</h2>
            <table>
                <tr><th>Reference</th><th>Name</th><th>Party</th><th>Date</th><th>Time</th><th>Phone</th><th>Status</th></tr>
                {"".join([f'<tr><td>{b.get("booking_reference", "")}</td><td>{b.get("name", "")}</td><td>{b.get("party_size", "")}</td><td>{b.get("date", "")}</td><td>{b.get("time", "")}</td><td>...{b.get("phone_number", "")[-4:] if b.get("phone_number") else ""}</td><td>{b.get("status", "")}</td></tr>' for b in bookings[-10:]])}
            </table>
            
            <h2>Recent Conversations (saved to conversations.json)</h2>
            {"".join([f'''
            <div class="conversation">
                <div class="user">Customer (...{c.get("phone_number", "")[-4:] if c.get("phone_number") else ""}): {c.get("message", "")}</div>
                <div class="bot">Bot: {c.get("response", "")}</div>
                <small>{c.get("timestamp", "")}</small>
            </div>
            ''' for c in conversations])}
            
            <p><strong>Files:</strong> Check bookings.json, conversations.json, restaurant_config.json, ./chroma_db/</p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/health")
async def health_check():
    restaurant_config = data_manager.load_restaurant_config()
    return {
        "status": "OK", 
        "version": "3.0-RAG-Test", 
        "timestamp": datetime.now().isoformat(),
        "menu_uploaded": restaurant_config.get('menu_uploaded', False),
        "rag_system": "active"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)