# Document Chatbot

This is the FastAPI backend for the Document Chatbot application.

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
- Windows: 
```
venv\Scripts\activate
```
- macOS/Linux: 
```
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```
export OPENAI_API_KEY=your_api_key_here
```
or on Windows:
```
set OPENAI_API_KEY=your_api_key_here
```

5. Run the server:
```
uvicorn main:app --reload
```

The API will be available at http://localhost:8000