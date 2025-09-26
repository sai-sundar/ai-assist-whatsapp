from fastapi import FastAPI, Form, Request
from fastapi.responses import Response, HTMLResponse
import sqlite3
from datetime import datetime
from typing import List, Dict
import os
import uvicorn
import re

# LangChain imports
from langchain.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
import json

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Restaurant Bot with LangChain", version="2.0.0")

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create memory storage for each user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memory (
            phone_number TEXT PRIMARY KEY,
            memory_data TEXT,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Restaurant information
RESTAURANT_INFO = """
Restaurant Information:
- Name: Bella Vista Restaurant
- Cuisine: Italian with local Luxembourg touches
- Hours: Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM, Closed Sundays
- Location: 15 Rue de la Paix, Luxembourg City
- Specialties: Homemade pasta, Wood-fired pizza, Local wine selection
- Vegetarian/Vegan: Yes, multiple options available
- Reservations: Recommended, especially weekends
- Average meal price: €25-35 per person
- Contact: +352 12 34 56 78
- Payment: Cash, cards accepted
"""

# Create conversational prompt template
CONVERSATION_TEMPLATE = """You are Maria, a friendly staff member at Bella Vista Restaurant in Luxembourg. You chat naturally like texting a friend on WhatsApp.

Restaurant Details:
- Name: Bella Vista Restaurant
- Cuisine: Italian with local Luxembourg touches
- Hours: Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM, Closed Sundays
- Location: 15 Rue de la Paix, Luxembourg City
- Specialties: Homemade pasta, Wood-fired pizza, Local wine selection
- Vegetarian/Vegan options available
- Contact: +352 12 34 56 78

PERSONALITY & STYLE:
- Be warm, friendly, and conversational
- Use casual language: "Hey!", "Awesome!", "Perfect!", emojis when appropriate
- Remember what customers tell you and refer back to it
- Chat like a real person, not a formal assistant
- Keep responses short and natural (1-2 sentences)

BOOKING HANDLING:
- When customers mention booking details, acknowledge everything they tell you
- If they give partial info, ask for missing details naturally
- For complete bookings, confirm everything and provide reference
- Be flexible - don't follow rigid scripts

EXAMPLES:
Customer: "Hi, do you have tables tonight?"
You: "Hey! Yeah we should have availability tonight. How many people?"

Customer: "Table for 4 at 8pm tomorrow"
You: "Perfect! Table for 4 tomorrow at 8pm. Just need your name to book it 😊"

Current conversation:
{history}
Customer: {input}
Maria:"""

class DatabaseManager:
    """Handle all database operations"""
    
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
    
    def save_user_memory(self, phone_number: str, memory_data: str):
        """Save user's conversation memory"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO user_memory (phone_number, memory_data, last_updated) VALUES (?, ?, ?)",
                (phone_number, memory_data, datetime.now())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Memory save error: {e}")
    
    def load_user_memory(self, phone_number: str) -> str:
        """Load user's conversation memory"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_data FROM user_memory WHERE phone_number = ?",
                (phone_number,)
            )
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else ""
        except Exception as e:
            print(f"Memory load error: {e}")
            return ""
    
    def save_booking(self, phone_number: str, booking_details: dict) -> str:
        """Save booking to database"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO bookings (phone_number, name, party_size, date, time) VALUES (?, ?, ?, ?, ?)",
                (phone_number, booking_details.get('name'), booking_details.get('party_size'), 
                 booking_details.get('date'), booking_details.get('time'))
            )
            conn.commit()
            booking_id = cursor.lastrowid
            conn.close()
            return f"BV{booking_id:03d}"
        except Exception as e:
            print(f"Booking error: {e}")
            return "BV999"
    
    def get_all_conversations(self, limit: int = 50) -> List[Dict]:
        """Get all conversations for admin dashboard"""
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
            print(f"Database error: {e}")
            return []
    
    def get_all_bookings(self) -> List[Dict]:
        """Get all bookings for admin dashboard"""
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
                    'timestamp': row[7]
                })
            return bookings
        except Exception as e:
            print(f"Database error: {e}")
            return []

class BookingExtractor:
    """Extract booking information from messages"""
    
    def extract_details(self, message: str) -> dict:
        """Extract booking information using regex patterns"""
        # Party size patterns
        party_patterns = [
            r'(?:table\s+)?(?:for\s+)?(\d{1,2})\s*(?:people|person|pax|guests?|adults?)',
            r'(\d{1,2})\s*(?:people|person|pax|guests?)',
            r'^\s*(\d{1,2})\s*$'  # Just a number
        ]
        
        party_size = None
        for pattern in party_patterns:
            match = re.search(pattern, message.lower())
            if match:
                party_size = int(match.group(1))
                break
        
        # Date patterns
        date_patterns = [
            r'\b(today|tomorrow|tonight)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})\b'
        ]
        
        date_match = None
        for pattern in date_patterns:
            match = re.search(pattern, message.lower())
            if match:
                date_match = match.group()
                break
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(?:am|pm)\b',
            r'\b(\d{1,2})\s*(?:am|pm)\b',
            r'\b(\d{1,2}):(\d{2})\b',
            r'\b(noon|midnight)\b'
        ]
        
        time_match = None
        for pattern in time_patterns:
            match = re.search(pattern, message.lower())
            if match:
                time_match = match.group()
                break
        
        # Name extraction (simple heuristic)
        name_match = None
        words = message.strip().split()
        if (len(words) <= 3 and 
            not party_size and 
            not date_match and 
            not time_match and
            not re.search(r'\b(table|book|reserve)\b', message.lower()) and
            len(message.strip()) > 2):
            name_match = message.strip()
        
        return {
            'party_size': party_size,
            'date': date_match,
            'time': time_match,
            'name': name_match
        }

class LangChainChatBot:
    """LangChain-powered chatbot with memory"""
    
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = Ollama(
            model="mistral:latest",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        # Create prompt template with correct variables
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=CONVERSATION_TEMPLATE + "\nCustomer: {input}\nMaria:"
        )
        
        # Store user conversations (phone_number -> ConversationChain)
        self.user_chains = {}
        
    def get_user_chain(self, phone_number: str) -> ConversationChain:
        """Get or create conversation chain for user"""
        if phone_number not in self.user_chains:
            # Create memory for this user
            memory = ConversationBufferWindowMemory(
                k=6,  # Remember last 6 exchanges
                return_messages=False
            )
            
            # Try to load previous memory from database
            db_manager = DatabaseManager()
            saved_memory = db_manager.load_user_memory(phone_number)
            if saved_memory:
                try:
                    # Restore memory from saved data
                    memory_dict = json.loads(saved_memory)
                    memory.chat_memory.messages = memory_dict.get('messages', [])
                except:
                    pass  # Start fresh if corrupted
            
            # Create conversation chain
            self.user_chains[phone_number] = ConversationChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=memory,
                verbose=False
            )
        
        return self.user_chains[phone_number]
    
    def chat(self, phone_number: str, message: str) -> str:
        """Chat with user maintaining conversation memory"""
        try:
            chain = self.get_user_chain(phone_number)
            
            # Generate response (LangChain handles history automatically)
            response = chain.predict(input=message)
            
            # Save memory to database
            memory_data = {
                'messages': [msg.dict() if hasattr(msg, 'dict') else str(msg) 
                           for msg in chain.memory.chat_memory.messages]
            }
            db_manager = DatabaseManager()
            db_manager.save_user_memory(phone_number, json.dumps(memory_data))
            
            return response.strip()
            
        except Exception as e:
            print(f"Chat error: {e}")
            return "Hey! Sorry, having some tech issues 😅 Can you try again or call us at +352 12 34 56 78?"

# Initialize components
db_manager = DatabaseManager()
booking_extractor = BookingExtractor()
chatbot = LangChainChatBot()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("🚀 LangChain Restaurant Bot started successfully!")
    print("📊 Admin dashboard: http://localhost:8000/admin")
    print("🏥 Health check: http://localhost:8000/health")

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    """Main WhatsApp webhook endpoint with LangChain"""
    try:
        message = Body.strip()
        phone_number = From
        
        print(f"📱 Message from {phone_number}: {message}")
        
        # Extract booking details
        booking_details = booking_extractor.extract_details(message)
        
        # Get AI response using LangChain
        ai_response = chatbot.chat(phone_number, message)
        
        # Check if we have complete booking info
        if all([booking_details.get('party_size'), booking_details.get('date'), 
                booking_details.get('time'), booking_details.get('name')]):
            booking_ref = db_manager.save_booking(phone_number, booking_details)
            print(f"✅ Booking saved: {booking_ref}")
            # Add reference to response if not already mentioned
            if "reference" not in ai_response.lower() and "ref" not in ai_response.lower():
                ai_response += f" Your reference is {booking_ref} ✨"
        
        # Log booking details if found
        if any(booking_details.values()):
            print(f"🔍 Extracted: {booking_details}")
        
        # Save conversation
        db_manager.save_conversation(phone_number, message, ai_response)
        
        # Return TwiML response
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{ai_response}</Message>
</Response>"""
        
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        error_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Hey! Something went wrong on our end 😅 Can you call us at +352 12 34 56 78?</Message>
</Response>"""
        return Response(content=error_response, media_type="application/xml")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "timestamp": datetime.now().isoformat(), "version": "2.0-LangChain"}

@app.get("/test-chat")
async def test_chat():
    """Test LangChain chatbot"""
    try:
        response = chatbot.chat("+1234567890", "Hi, what are your opening hours?")
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Enhanced admin dashboard"""
    conversations = db_manager.get_all_conversations()
    bookings = db_manager.get_all_bookings()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🍽️ Bella Vista Bot Admin (LangChain)</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                color: white;
            }}
            .version {{
                background: #28a745;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.8em;
                margin-left: 10px;
            }}
            .stats {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .stat-card {{ 
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                color: white; 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .stat-number {{ 
                font-size: 2.5em; 
                font-weight: bold; 
                margin-bottom: 5px;
            }}
            .conversation {{ 
                margin-bottom: 20px; 
                padding: 20px; 
                border: none; 
                border-radius: 15px; 
                background: #f8f9fa;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .user {{ 
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                color: white;
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 10px;
            }}
            .bot {{ 
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                color: white;
                padding: 15px; 
                border-radius: 10px; 
                margin-bottom: 10px;
            }}
            .timestamp {{ 
                color: #6c757d; 
                font-size: 0.9em; 
                text-align: right;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            th {{
                background: linear-gradient(135deg, #6f42c1 0%, #5a2d91 100%);
                color: white;
                padding: 15px;
                font-weight: 600;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            tr:hover {{
                background: #f8f9fa;
            }}
            h2 {{
                color: #495057;
                margin-top: 40px;
                border-left: 5px solid #007bff;
                padding-left: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍽️ Bella Vista Restaurant Bot Admin</h1>
                <span class="version">LangChain v2.0</span>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(conversations)}</div>
                    <div>Total Conversations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(bookings)}</div>
                    <div>Total Bookings</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(set(conv['phone_number'] for conv in conversations))}</div>
                    <div>Unique Customers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">🧠</div>
                    <div>Memory Enabled</div>
                </div>
            </div>
            
            <h2>📝 Recent Bookings</h2>
            <table>
                <tr>
                    <th>📱 Phone</th>
                    <th>👤 Name</th>
                    <th>👥 Party</th>
                    <th>📅 Date</th>
                    <th>⏰ Time</th>
                    <th>✅ Status</th>
                    <th>🕒 Timestamp</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{booking["phone_number"][-4:]}</td>
                    <td>{booking["name"] or "N/A"}</td>
                    <td>{booking["party_size"] or "N/A"}</td>
                    <td>{booking["date"] or "N/A"}</td>
                    <td>{booking["time"] or "N/A"}</td>
                    <td><span style="color: #28a745;">●</span> {booking["status"]}</td>
                    <td>{booking["timestamp"]}</td>
                </tr>
                ''' for booking in bookings])}
            </table>
            
            <h2>💬 Recent Conversations (with Memory)</h2>
            {"".join([f'''
            <div class="conversation">
                <div class="user">
                    <strong>🙋‍♂️ Customer (...{conv["phone_number"][-4:]}):</strong> {conv["message"]}
                </div>
                <div class="bot">
                    <strong>🤖 Maria:</strong> {conv["response"]}
                </div>
                <div class="timestamp">📅 {conv["timestamp"]}</div>
            </div>
            ''' for conv in conversations[:15]])}
            
            <div style="text-align: center; margin-top: 30px; color: #6c757d;">
                <p>🧠 Powered by LangChain with Conversation Memory</p>
                <p>🚀 Each customer's conversation is remembered across sessions</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)