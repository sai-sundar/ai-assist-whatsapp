# Complete WhatsApp Restaurant Bot with LangGraph + LangChain
# File: main.py

from fastapi import FastAPI, Form, Request
from fastapi.responses import Response, HTMLResponse
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Annotated, Literal
import os
import uvicorn
import re
import json

# LangChain imports
from langchain_community.llms import Ollama
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Restaurant Bot with LangGraph", version="3.0.0")

# Database setup
DATABASE_FILE = "restaurant_bot.db"

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            message TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create bookings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            name TEXT,
            party_size INTEGER,
            date TEXT,
            time TEXT,
            status TEXT DEFAULT 'confirmed',
            special_requests TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Restaurant information
RESTAURANT_INFO = """
Bella Vista Restaurant - Italian with Luxembourg touches
Hours: Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM, Closed Sundays
Location: 15 Rue de la Paix, Luxembourg City
Phone: +352 12 34 56 78
Specialties: Homemade pasta, Wood-fired pizza, Local wine
"""

# Define the conversation state
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    phone_number: str
    current_intent: str
    booking_data: dict
    next_action: str

# Define tools
class CreateBookingTool(BaseTool):
    name = "create_booking"
    description = "Create a restaurant reservation with all required details"
    
    def _run(self, booking_data: str) -> str:
        """Execute booking creation"""
        try:
            # Parse booking data (expected as JSON string)
            data = json.loads(booking_data)
            
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bookings (phone_number, name, party_size, date, time, special_requests) VALUES (?, ?, ?, ?, ?, ?)",
                (data.get("phone_number", ""), data.get("name", ""), data.get("party_size", 2), 
                 data.get("date", ""), data.get("time", ""), data.get("special_requests", ""))
            )
            conn.commit()
            booking_id = cursor.lastrowid
            conn.close()
            
            booking_ref = f"BV{booking_id:03d}"
            print(f"✅ Booking created: {booking_ref}")
            
            return f"Booking confirmed! Reference: {booking_ref}. Table for {data.get('party_size')} people on {data.get('date')} at {data.get('time')} under {data.get('name')}."
            
        except Exception as e:
            print(f"Booking tool error: {e}")
            return "Sorry, I had trouble creating the booking. Please try again."

class CheckAvailabilityTool(BaseTool):
    name = "check_availability"
    description = "Check table availability for a specific date and time"
    
    def _run(self, date_time: str) -> str:
        """Check availability"""
        try:
            # Simple availability logic
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM bookings WHERE date LIKE ? AND time LIKE ?",
                (f"%{date_time}%", f"%{date_time}%")
            )
            count = cursor.fetchone()[0]
            conn.close()
            
            available_tables = 20 - count
            
            if available_tables > 5:
                return f"Great availability! Plenty of tables available."
            elif available_tables > 0:
                return f"Limited availability but we can accommodate you."
            else:
                return f"Sorry, we're fully booked. Can I suggest alternative times?"
                
        except Exception as e:
            return "I can check availability - just let me know your preferred date and time."

# Initialize tools
tools = [CreateBookingTool(), CheckAvailabilityTool()]

# Initialize LLM
llm = Ollama(
    model="mistral:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Graph node functions
def classify_intent(state: ConversationState) -> ConversationState:
    """Classify user intent from their message"""
    if not state["messages"]:
        return state
    
    last_message = state["messages"][-1].content.lower()
    booking_data = state.get("booking_data", {})
    
    # Check if we have an ongoing booking with missing information
    if booking_data:
        required_fields = ["name", "party_size", "date", "time"]
        missing = [f for f in required_fields if f not in booking_data or not booking_data[f]]
        
        if missing:
            print(f"🔄 Continuing existing booking, missing: {missing}")
            intent = "booking"  # Continue booking flow
        else:
            # Check for new booking intent
            if any(word in last_message for word in ["book", "reserve", "table", "reservation"]):
                intent = "booking"
            else:
                intent = "general_chat"
    else:
        # No existing booking, classify normally
        if any(word in last_message for word in ["book", "reserve", "table", "reservation"]):
            intent = "booking"
        elif any(word in last_message for word in ["menu", "food", "dish", "special"]):
            intent = "menu_inquiry"
        elif any(word in last_message for word in ["hours", "open", "close", "time"]):
            intent = "hours_inquiry"
        elif any(word in last_message for word in ["location", "address", "where"]):
            intent = "location_inquiry"
        else:
            intent = "general_chat"
    
    print(f"🔍 Intent classified: {intent} (existing booking: {bool(booking_data)})")
    
    return {
        **state,
        "current_intent": intent,
        "next_action": "route_conversation"
    }

def route_conversation(state: ConversationState) -> Literal["handle_booking", "provide_info", "general_chat"]:
    """Route conversation based on intent"""
    intent = state["current_intent"]
    
    if intent == "booking":
        return "handle_booking"
    elif intent in ["menu_inquiry", "hours_inquiry", "location_inquiry"]:
        return "provide_info"
    else:
        return "general_chat"

def handle_booking(state: ConversationState) -> ConversationState:
    """Handle booking conversation flow"""
    messages = state["messages"]
    booking_data = state.get("booking_data", {})
    
    # Get the last HUMAN message, not the AI message
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not human_messages:
        return state
        
    last_human_message = human_messages[-1].content
    
    print(f"🔍 Processing human message: '{last_human_message}'")
    
    # Extract information from current message
    extracted = extract_booking_details(last_human_message)
    booking_data.update(extracted)
    
    print(f"🔍 Extracted booking data: {extracted}")
    print(f"📋 Current booking data: {booking_data}")
    
    # Check what we still need
    required_fields = ["name", "party_size", "date", "time"]
    missing = [f for f in required_fields if f not in booking_data or not booking_data[f]]
    
    print(f"❓ Missing fields: {missing}")
    
    if not missing:
        # Complete booking - create it
        booking_json = json.dumps({
            **booking_data,
            "phone_number": state["phone_number"]
        })
        
        result = tools[0]._run(booking_json)
        response = f"Perfect! {result}"
        next_action = "end"  # CRITICAL: Set to end to stop recursion
    else:
        # Ask for missing information
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
        
        next_action = "end"  # ALWAYS end after asking a question, wait for user response
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "booking_data": booking_data,
        "next_action": next_action
    }

def provide_info(state: ConversationState) -> ConversationState:
    """Provide restaurant information"""
    intent = state["current_intent"]
    
    if intent == "menu_inquiry":
        response = "Our specialties include homemade pasta, wood-fired pizza, and local wine selection! We have excellent vegetarian and vegan options too. Would you like to make a reservation?"
    elif intent == "hours_inquiry":
        response = "We're open Monday-Thursday 11:30AM-10PM, Friday-Saturday 11:30AM-11PM. Closed Sundays. Would you like to book a table?"
    elif intent == "location_inquiry":
        response = "You can find us at 15 Rue de la Paix, Luxembourg City. We're in the heart of the city! Need a reservation?"
    else:
        response = "I'm here to help with any questions about Bella Vista Restaurant!"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "end"
    }

def general_chat(state: ConversationState) -> ConversationState:
    """Handle general conversation"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Check if they want to make a reservation
    if any(word in last_message.lower() for word in ["reservation", "book", "table"]):
        response = """I'd be happy to help you make a reservation! You can either:

📝 Tell me your details naturally, or
📋 Use this quick format:

*Name: [Your name]*
*Party size: [Number of people]*
*Date: [Date you prefer]*  
*Time: [Time you prefer]*

Just copy and fill in your details!"""
    else:
        # Create a simple prompt for general conversation
        prompt = f"""You are Maria, a friendly staff member at Bella Vista Restaurant in Luxembourg.

{RESTAURANT_INFO}

Customer says: {last_message}

Respond warmly and naturally (1-2 sentences). Try to guide towards making a reservation when appropriate."""
        
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            response = "Hi! I'm here to help with any questions about Bella Vista Restaurant. Would you like to make a reservation?"
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "next_action": "end"
    }

def extract_booking_details(text: str) -> dict:
    """Extract booking information from text"""
    details = {}
    text_lower = text.lower()
    
    print(f"🔍 Extracting from: '{text}'")
    
    # Check for structured template format first
    template_patterns = {
        'name': r'name:\s*([^\n\r*]+)',
        'party_size': r'party\s+size:\s*(\d+)',
        'date': r'date:\s*([^\n\r*]+)',
        'time': r'time:\s*([^\n\r*]+)'
    }
    
    for field, pattern in template_patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).strip()
            if field == 'party_size':
                details[field] = int(value)
            else:
                details[field] = value
            print(f"   Found {field} (template): {details[field]}")
    
    # If we found template format, return early
    if details:
        print(f"🔍 Template extraction complete: {details}")
        return details
    
    # Extract party size - more comprehensive patterns
    party_patterns = [
        r'party\s+of\s+(\d{1,2})',  # "party of 4"
        r'(?:for\s+)?(\d{1,2})\s*(?:people|person|pax|guests?)',
        r'(\d{1,2})\s*(?:people|person|pax|guests?)',
        r'table\s+for\s+(\d{1,2})',
        r'(\d{1,2})\s+(?:people|guests?|diners?)'
    ]
    
    for pattern in party_patterns:
        match = re.search(pattern, text_lower)
        if match:
            details["party_size"] = int(match.group(1))
            print(f"   Found party size: {details['party_size']}")
            break
    
    # Extract date - handle "October 4" format
    date_patterns = [
        r'\b(today|tomorrow|tonight)\b',
        r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b',
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(\d{1,2})[/-](\d{1,2})\b'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:
                # Handle "October 4" or "4 October" format
                details["date"] = f"{match.group(1)} {match.group(2)}"
            else:
                details["date"] = match.group()
            print(f"   Found date: {details['date']}")
            break
    
    # Extract time - handle "11:30 am" format
    time_patterns = [
        r'\b(\d{1,2}):(\d{2})\s*(?:am|pm)\b',
        r'\b(\d{1,2}):(\d{2})\b',
        r'\b(\d{1,2})\s*(?:am|pm)\b',
        r'\b(noon|midnight)\b'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if len(match.groups()) == 2:
                # Handle "11:30 am" format
                details["time"] = f"{match.group(1)}:{match.group(2)}"
                # Find am/pm separately
                ampm_match = re.search(r'\b(?:am|pm)\b', text_lower)
                if ampm_match:
                    details["time"] += f" {ampm_match.group()}"
            else:
                details["time"] = match.group()
            print(f"   Found time: {details['time']}")
            break
    
    # Smart name extraction - much more liberal
    words = text.strip().split()
    
    # If it's a short message with no booking keywords and no numbers, it's likely a name
    booking_keywords = ['table', 'book', 'reserve', 'party', 'people', 'reservation', 'october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    has_booking_keywords = any(keyword in text_lower for keyword in booking_keywords)
    has_numbers = any(char.isdigit() for char in text)
    
    # If it's 1-4 words, no booking keywords, and not obviously not a name
    if (len(words) <= 4 and 
        not has_booking_keywords and
        not has_numbers and
        len(text.strip()) > 1 and
        not any(details.values())):  # Only extract name if we didn't find other booking details
        
        # Additional checks: not common phrases
        common_phrases = ['yes', 'no', 'ok', 'okay', 'thanks', 'thank you', 'hello', 'hi', 'hey']
        if text_lower not in common_phrases:
            details["name"] = text.strip()
            print(f"   Found name: {details['name']}")
    
    print(f"🔍 Extracted details: {details}")
    return details

# Build the conversation graph
def build_conversation_graph():
    """Build the LangGraph conversation flow"""
    
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_booking", handle_booking)
    workflow.add_node("provide_info", provide_info)
    workflow.add_node("general_chat", general_chat)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify_intent",
        route_conversation,
        {
            "handle_booking": "handle_booking",
            "provide_info": "provide_info",
            "general_chat": "general_chat"
        }
    )
    
    # Add simple edges - no conditional routing for booking continuation
    workflow.add_edge("handle_booking", END)
    workflow.add_edge("provide_info", END)
    workflow.add_edge("general_chat", END)
    
    # Compile with memory only
    memory = SqliteSaver.from_conn_string(":memory:")
    return workflow.compile(checkpointer=memory)

# Initialize the conversation graph
conversation_graph = build_conversation_graph()

class DatabaseManager:
    """Handle database operations"""
    
    def save_conversation(self, phone_number: str, message: str, response: str):
        """Save conversation to database"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (phone_number, message, response) VALUES (?, ?, ?)",
                (phone_number, message, response)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def get_bookings_for_date(self, date: str) -> List[Dict]:
        """Get all bookings for a specific date"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM bookings WHERE date LIKE ? ORDER BY time",
                (f"%{date}%",)
            )
            rows = cursor.fetchall()
            conn.close()
            
            bookings = []
            for row in rows:
                bookings.append({
                    'id': row[0],
                    'phone_number': row[1],
                    'name': row[2],
                    'party_size': row[3],
                    'date': row[4],
                    'time': row[5],
                    'status': row[6],
                    'special_requests': row[7],
                    'timestamp': row[8]
                })
            return bookings
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
    def get_all_conversations(self, limit: int = 50) -> List[Dict]:
        """Get all conversations"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()
            
            conversations = []
            for row in rows:
                conversations.append({
                    'id': row[0],
                    'phone_number': row[1],
                    'message': row[2],
                    'response': row[3],
                    'timestamp': row[4]
                })
            return conversations
        except Exception as e:
            return []
    
    def get_all_bookings(self) -> List[Dict]:
        """Get all bookings"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bookings ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            conn.close()
            
            bookings = []
            for row in rows:
                bookings.append({
                    'id': row[0],
                    'phone_number': row[1],
                    'name': row[2],
                    'party_size': row[3],
                    'date': row[4],
                    'time': row[5],
                    'status': row[6],
                    'special_requests': row[7],
                    'timestamp': row[8]
                })
            return bookings
        except Exception as e:
            return []

# Initialize components
db_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("🚀 LangGraph Restaurant Bot started!")
    print("📊 Admin: http://localhost:8000/admin")
    print("📅 Bookings: http://localhost:8000/bookings")

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    """Main WhatsApp webhook with LangGraph"""
    try:
        message = Body.strip()
        phone_number = From
        
        print(f"📱 Message from {phone_number}: {message}")
        
        # Create initial state
        initial_state = ConversationState(
            messages=[HumanMessage(content=message)],
            phone_number=phone_number,
            current_intent="",
            booking_data={},
            next_action=""
        )
        
        # Run through conversation graph with recursion limit
        config = {
            "configurable": {"thread_id": phone_number},
            "recursion_limit": 50
        }
        result = conversation_graph.invoke(initial_state, config)
        
        # Get AI response
        ai_response = result["messages"][-1].content
        
        # Save to database
        db_manager.save_conversation(phone_number, message, ai_response)
        
        print(f"🤖 Response: {ai_response}")
        
        # Return TwiML
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{ai_response}</Message>
</Response>"""
        
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        error_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sorry, something went wrong. Please call +352 12 34 56 78</Message>
</Response>"""
        return Response(content=error_response, media_type="application/xml")

@app.get("/health")
async def health_check():
    return {"status": "OK", "version": "3.0-LangGraph", "timestamp": datetime.now().isoformat()}

@app.get("/test-graph")
async def test_graph():
    """Test the conversation graph"""
    try:
        test_state = ConversationState(
            messages=[HumanMessage(content="I need a table for 4 people")],
            phone_number="+1234567890",
            current_intent="",
            booking_data={},
            next_action=""
        )
        
        config = {"configurable": {"thread_id": "test"}}
        result = conversation_graph.invoke(test_state, config)
        
        return {"success": True, "response": result["messages"][-1].content}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/bookings", response_class=HTMLResponse)
async def bookings_dashboard():
    """Bookings dashboard"""
    today_bookings = db_manager.get_bookings_for_date("today")
    tomorrow_bookings = db_manager.get_bookings_for_date("tomorrow")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📅 Bookings Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #f8f9fa; }}
            .no-bookings {{ text-align: center; color: #999; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📅 Bella Vista Bookings Dashboard</h1>
                <p>LangGraph-Powered System</p>
            </div>
            
            <div class="section">
                <h2>Today's Bookings ({len(today_bookings)})</h2>
                {('<div class="no-bookings">No bookings today</div>' if not today_bookings else 
                f'''<table>
                    <tr><th>Time</th><th>Name</th><th>Party</th><th>Phone</th><th>Status</th></tr>
                    {"".join([f'<tr><td>{b["time"]}</td><td>{b["name"]}</td><td>{b["party_size"]}</td><td>...{b["phone_number"][-4:]}</td><td>{b["status"]}</td></tr>' for b in today_bookings])}
                </table>''')}
            </div>
            
            <div class="section">
                <h2>Tomorrow's Bookings ({len(tomorrow_bookings)})</h2>
                {('<div class="no-bookings">No bookings tomorrow</div>' if not tomorrow_bookings else 
                f'''<table>
                    <tr><th>Time</th><th>Name</th><th>Party</th><th>Phone</th><th>Status</th></tr>
                    {"".join([f'<tr><td>{b["time"]}</td><td>{b["name"]}</td><td>{b["party_size"]}</td><td>...{b["phone_number"][-4:]}</td><td>{b["status"]}</td></tr>' for b in tomorrow_bookings])}
                </table>''')}
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard"""
    conversations = db_manager.get_all_conversations()
    bookings = db_manager.get_all_bookings()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🍽️ Admin Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #fd7e14, #e55d3b); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background: #f8f9fa; }}
            .conversation {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .user {{ background: #007bff; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
            .bot {{ background: #28a745; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍽️ Bella Vista Admin Dashboard</h1>
                <p>LangGraph v3.0</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(conversations)}</div>
                    <div>Conversations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(bookings)}</div>
                    <div>Bookings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(set(c['phone_number'] for c in conversations))}</div>
                    <div>Unique Customers</div>
                </div>
            </div>
            
            <h2>Recent Bookings</h2>
            <table>
                <tr><th>Name</th><th>Party</th><th>Date</th><th>Time</th><th>Status</th><th>Phone</th></tr>
                {"".join([f'<tr><td>{b["name"]}</td><td>{b["party_size"]}</td><td>{b["date"]}</td><td>{b["time"]}</td><td>{b["status"]}</td><td>...{b["phone_number"][-4:]}</td></tr>' for b in bookings[:10]])}
            </table>
            
            <h2>Recent Conversations</h2>
            {"".join([f'''
            <div class="conversation">
                <div class="user">Customer (...{c["phone_number"][-4:]}): {c["message"]}</div>
                <div class="bot">Bot: {c["response"]}</div>
                <small>{c["timestamp"]}</small>
            </div>
            ''' for c in conversations[:10]])}
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)