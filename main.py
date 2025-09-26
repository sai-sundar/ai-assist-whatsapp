# WhatsApp Restaurant Bot MVP with Ollama + Mistral 7B (Python)
# File: main.py

from fastapi import FastAPI, Form, Request
from fastapi.responses import Response, HTMLResponse
import sqlite3
import requests
import json
from datetime import datetime
from typing import List, Dict
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Restaurant Bot", version="1.0.0")

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
            status TEXT DEFAULT 'pending',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Restaurant information for the AI
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

SYSTEM_PROMPT = f"""You are a helpful restaurant assistant for Bella Vista Restaurant in Luxembourg. 
You help customers with:
1. Answering questions about the restaurant (menu, hours, location, etc.)
2. Taking table reservations
3. Providing information about food options

Key behaviors:
- Be friendly and professional
- Always respond in the same language the customer uses
- For bookings, ask for: name, party size, preferred date, and time
- If customer wants to book, ask for their details step by step
- Keep responses concise but helpful (max 2-3 sentences)
- If you don't know something, politely say so and offer to connect them with staff

Restaurant info: {RESTAURANT_INFO}

Always be helpful and try to convert inquiries into reservations when appropriate."""

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "mistral:latest"
    
    async def chat(self, message: str, conversation_history: List[Dict] = None) -> str:
        """Send message to Ollama and get response"""
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history for context
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return self._get_fallback_response()
                
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response when AI is unavailable"""
        return ("I apologize, but I'm having trouble processing your request right now. "
                "Please try again in a moment or call us directly at +352 12 34 56 78.")

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
    
    def get_conversation_history(self, phone_number: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history for context"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message, response FROM conversations WHERE phone_number = ? ORDER BY timestamp DESC LIMIT ?",
                (phone_number, limit)
            )
            rows = cursor.fetchall()
            conn.close()
            
            # Format for AI context (reverse order for chronological)
            history = []
            for row in reversed(rows):
                history.append({"role": "user", "content": row[0]})
                history.append({"role": "assistant", "content": row[1]})
            
            return history
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
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

# Initialize components
ollama_client = OllamaClient()
db_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    print("🚀 Restaurant Bot started successfully!")
    print("📊 Admin dashboard: http://localhost:8000/admin")
    print("🏥 Health check: http://localhost:8000/health")

@app.post("/webhook")
async def webhook(Body: str = Form(...), From: str = Form(...)):
    """Main WhatsApp webhook endpoint"""
    try:
        message = Body.strip()
        phone_number = From
        
        print(f"📱 Received message from {phone_number}: {message}")
        
        # Get conversation history for context
        history = db_manager.get_conversation_history(phone_number)
        
        # Get AI response
        ai_response = await ollama_client.chat(message, history)
        
        # Save conversation
        db_manager.save_conversation(phone_number, message, ai_response)
        
        # Check for booking keywords (simple detection for demo)
        booking_keywords = ['book', 'reserve', 'table', 'reservation']
        if any(keyword in message.lower() for keyword in booking_keywords):
            print("🍽️ Potential booking detected!")
        
        # Return TwiML response for WhatsApp
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{ai_response}</Message>
</Response>"""
        
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        error_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Sorry, I'm having technical difficulties. Please call us at +352 12 34 56 78.</Message>
</Response>"""
        return Response(content=error_response, media_type="application/xml")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "timestamp": datetime.now().isoformat()}

@app.get("/test-ollama")
async def test_ollama():
    """Test Ollama connection"""
    try:
        response = await ollama_client.chat("Hello, what are your restaurant hours?")
        return {"success": True, "response": response}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard to view conversations and bookings"""
    conversations = db_manager.get_all_conversations()
    bookings = db_manager.get_all_bookings()
    
    # Generate HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🍽️ Bella Vista Bot Admin</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .stats {{ 
                display: flex; 
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .stat-card {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                flex: 1; 
                text-align: center;
            }}
            .stat-number {{ 
                font-size: 2em; 
                font-weight: bold; 
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0; 
                background: white;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            th {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                font-weight: 600;
            }}
            .conversation {{ 
                margin-bottom: 15px; 
                padding: 15px; 
                border: 1px solid #eee; 
                border-radius: 10px; 
                background: #fafafa;
            }}
            .user {{ 
                background-color: #e3f2fd; 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 5px;
            }}
            .bot {{ 
                background-color: #f3e5f5; 
                padding: 10px; 
                border-radius: 5px; 
                margin-bottom: 5px;
            }}
            .timestamp {{ 
                color: #666; 
                font-size: 0.9em; 
            }}
            h1 {{ 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px;
            }}
            h2 {{ 
                color: #555; 
                border-bottom: 2px solid #667eea; 
                padding-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍽️ Bella Vista Restaurant Bot Admin</h1>
            
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
            </div>
            
            <h2>📝 Recent Bookings</h2>
            <table>
                <tr>
                    <th>📱 Phone</th>
                    <th>👤 Name</th>
                    <th>👥 Party Size</th>
                    <th>📅 Date</th>
                    <th>⏰ Time</th>
                    <th>✅ Status</th>
                    <th>🕒 Timestamp</th>
                </tr>
                {"".join([f'''
                <tr>
                    <td>{booking["phone_number"]}</td>
                    <td>{booking["name"] or "N/A"}</td>
                    <td>{booking["party_size"] or "N/A"}</td>
                    <td>{booking["date"] or "N/A"}</td>
                    <td>{booking["time"] or "N/A"}</td>
                    <td>{booking["status"]}</td>
                    <td>{booking["timestamp"]}</td>
                </tr>
                ''' for booking in bookings])}
            </table>
            
            <h2>💬 Recent Conversations</h2>
            {"".join([f'''
            <div class="conversation">
                <div class="user">
                    <strong>🙋‍♂️ Customer ({conv["phone_number"]}):</strong> {conv["message"]}
                </div>
                <div class="bot">
                    <strong>🤖 Bot:</strong> {conv["response"]}
                </div>
                <div class="timestamp">📅 {conv["timestamp"]}</div>
            </div>
            ''' for conv in conversations[:20]])}
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)