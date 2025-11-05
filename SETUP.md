# Setup Guide

Complete step-by-step guide for setting up the AI-Assisted WhatsApp Restaurant Bot.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Ollama Installation](#ollama-installation)
3. [Python Environment Setup](#python-environment-setup)
4. [Project Installation](#project-installation)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements:
- **OS**: Linux, macOS, or Windows 10/11
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum (16GB recommended for smooth LLM inference)
- **Disk**: 10GB free space (for models and vector database)
- **Python**: 3.8 or higher

### Software Dependencies:
- Python 3.8+
- pip (Python package manager)
- Git (for cloning repository)
- Ollama (for local LLM)

## Ollama Installation

Ollama is required for running the local LLM (Mistral model).

### macOS

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### Linux

```bash
# Download and install
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &
```

### Windows

1. Download the installer from https://ollama.com/download
2. Run the installer
3. Ollama will start automatically as a service

### Pull Required Models

After installing Ollama, pull the Mistral model:

```bash
# Pull Mistral 7B model (required for the bot)
ollama pull mistral:7b

# Verify the model is available
ollama list
```

**Note**: The first model pull will download ~4GB. Ensure stable internet connection.

### Verify Ollama is Running

```bash
# Check if Ollama is accessible
curl http://localhost:11434

# Test model inference
ollama run mistral:7b "Hello, how are you?"
```

## Python Environment Setup

### macOS/Linux

```bash
# Check Python version (must be 3.8+)
python3 --version

# If Python is not installed or version is too old:
# macOS: Install via Homebrew
brew install python3

# Ubuntu/Debian:
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora/RHEL:
sudo dnf install python3 python3-pip
```

### Windows

1. Download Python from https://python.org/downloads
2. Run installer and check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

## Project Installation

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/sai-sundar/ai-assist-whatsapp.git

# Navigate to project directory
cd ai-assist-whatsapp
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv)
```

**Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Your prompt should now show (venv)
```

**Note**: Always activate the virtual environment before running the application.

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install RAG and LangChain dependencies
pip install langchain langchain-community langgraph chromadb pypdf2 ollama

# Install Ollama embeddings
pip install langchain-ollama
```

**Expected Installation Time**: 2-5 minutes depending on internet speed.

### 4. Verify Installation

```bash
# Check installed packages
pip list | grep -E "langchain|fastapi|chromadb|ollama"
```

You should see:
- fastapi
- uvicorn
- langchain
- langchain-community
- langgraph
- chromadb
- pypdf2
- ollama (Python client)

## Configuration

### 1. Restaurant Configuration

The application creates a default `restaurant_config.json` on first run. To customize:

```bash
# Edit restaurant_config.json after first run
nano restaurant_config.json
```

Example configuration:
```json
{
  "name": "Bella Vista Restaurant",
  "cuisine_type": "Italian with Luxembourg touches",
  "hours": "Mon-Thu 11:30AM-10PM, Fri-Sat 11:30AM-11PM, Closed Sundays",
  "location": "15 Rue de la Paix, Luxembourg City",
  "phone": "+352 12 34 56 78",
  "max_guests": 20,
  "policies": "Cancellation required 2 hours in advance",
  "menu_uploaded": false
}
```

### 2. Ollama Configuration

If Ollama is not running on default port (11434), update in `main.py`:

```python
llm = Ollama(
    model="mistral:7b",
    base_url="http://localhost:11434",  # Change if different
    temperature=0.7
)

embeddings = OllamaEmbeddings(
    model="mistral:latest",
    base_url="http://localhost:11434"  # Change if different
)
```

### 3. File Permissions

Ensure the application can write to the project directory:

```bash
# macOS/Linux
chmod -R 755 .

# Create required directories (done automatically on startup)
mkdir -p chroma_db uploaded_menus
```

## Running the Application

### 1. Start Ollama (if not running)

**macOS/Linux:**
```bash
# Check if running
curl http://localhost:11434

# If not running:
ollama serve &
```

**Windows:**
- Ollama runs as a service automatically
- Check system tray for Ollama icon

### 2. Start the Application

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Run the application
python main.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
RAG system initialized successfully
🚀 RAG Test System started!
🧪 Test interface: http://localhost:8000/test
📤 Upload menu: http://localhost:8000/upload-menu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Access the Application

Open your browser and navigate to:
- **Test Interface**: http://localhost:8000/test
- **Admin Dashboard**: http://localhost:8000/admin
- **Health Check**: http://localhost:8000/health

### 4. Upload a Menu (Optional)

1. Navigate to http://localhost:8000/test
2. Click "Choose File" under "Upload Menu (PDF)"
3. Select a menu PDF file
4. Click "Upload Menu"
5. Wait for processing (may take 10-30 seconds for large menus)

**Note**: Menu PDF must contain extractable text. Image-only PDFs will fail.

## Troubleshooting

### Issue: "RAG system initialization error"

**Cause**: Ollama is not running or Mistral model is not available.

**Solution**:
```bash
# Check Ollama status
curl http://localhost:11434

# Start Ollama if not running
ollama serve &

# Pull Mistral model
ollama pull mistral:7b
```

### Issue: "No module named 'langchain'"

**Cause**: Dependencies not installed correctly.

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install langchain langchain-community langgraph chromadb
```

### Issue: "Address already in use" (Port 8000)

**Cause**: Another application is using port 8000.

**Solution**:
```bash
# Option 1: Kill the process using port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Option 2: Run on different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Issue: PDF upload fails with "No text extracted"

**Cause**: PDF contains images instead of text, or is password-protected.

**Solutions**:
1. Use a different PDF with extractable text
2. Convert image-based PDF using OCR (e.g., Adobe Acrobat, online tools)
3. Manually create a text version of the menu

### Issue: Slow LLM responses

**Cause**: Insufficient system resources or large model.

**Solutions**:
1. Close other applications to free RAM
2. Use a smaller model (if available):
   ```bash
   ollama pull mistral:7b-q4  # Quantized version (faster, slightly less quality)
   ```
3. Increase timeout in `main.py`:
   ```python
   llm = Ollama(
       model="mistral:7b",
       base_url="http://localhost:11434",
       temperature=0.7,
       timeout=120  # Increase timeout
   )
   ```

### Issue: "ChromaDB persistence error"

**Cause**: Permissions issue or corrupted vector database.

**Solution**:
```bash
# Delete and reinitialize ChromaDB
rm -rf chroma_db
# Restart application (it will recreate the directory)
python main.py
# Re-upload menu PDF
```

### Issue: Virtual environment not activating

**Windows PowerShell**:
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\Activate.ps1
```

### Issue: "ModuleNotFoundError: No module named 'uvicorn'"

**Cause**: requirements.txt not fully installed.

**Solution**:
```bash
# Install all requirements explicitly
pip install fastapi "uvicorn[standard]" python-multipart requests
pip install langchain langchain-community langgraph chromadb pypdf2 ollama
```

## Verification Checklist

Before running the application, ensure:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Virtual environment created and activated (`which python` shows venv path)
- [ ] All dependencies installed (`pip list`)
- [ ] Ollama installed and running (`curl http://localhost:11434`)
- [ ] Mistral model downloaded (`ollama list`)
- [ ] Port 8000 available (`lsof -ti:8000` returns nothing)
- [ ] Write permissions in project directory

## Performance Optimization

### For Better Response Times:

1. **Use GPU Acceleration** (if available):
   - Install CUDA/ROCm drivers
   - Ollama will automatically use GPU

2. **Increase RAM for Ollama**:
   ```bash
   # macOS/Linux: Set environment variable
   export OLLAMA_NUM_GPU=1
   export OLLAMA_MAX_LOADED_MODELS=1
   ```

3. **Use Quantized Models**:
   ```bash
   ollama pull mistral:7b-q4_0  # 4-bit quantization
   ```
   Update `main.py` to use the quantized model.

## Next Steps

After successful setup:

1. Test the chat interface at http://localhost:8000/test
2. Upload a sample menu PDF
3. Try example conversations (menu queries, bookings)
4. Check the admin dashboard at http://localhost:8000/admin
5. Review logs for any errors

## Getting Help

If you encounter issues not covered here:

1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for system details
2. Review application logs in terminal
3. Open an issue on GitHub with:
   - Operating system and version
   - Python version
   - Error messages and logs
   - Steps to reproduce

## Development Mode

For development with auto-reload:

```bash
# Run with auto-reload on code changes
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run with detailed logging
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## Production Deployment

For production use:

1. Use a proper WSGI server (e.g., Gunicorn)
2. Set up HTTPS with SSL certificates
3. Use a reverse proxy (e.g., Nginx)
4. Implement rate limiting and authentication
5. Use a production database (PostgreSQL)
6. Set up monitoring and logging
7. Configure environment variables for secrets

**Note**: This is a test/development system. Significant changes needed for production.
