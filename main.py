import gradio as gr
from transformers import pipeline
import sympy
import re
import ast
import json
from pathlib import Path
from datetime import datetime

# Initialize the chatbot model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Knowledge base for common queries
KNOWLEDGE_BASE = {
    "ai": "Artificial Intelligence is the simulation of human intelligence processes by machines.",
    "python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
    "gradio": "Gradio is a Python library that allows you to quickly create user interfaces for machine learning models.",
    "regex": "Regex, or Regular Expressions, is a sequence of characters that form a search pattern. It's used for string matching.",
    "machine learning": "Machine learning is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions based on data.",
    "deep learning": "Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data.",
    "data science": "Data science is a field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
}

# Session storage directory
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)

# Safe math evaluation using sympy
def evaluate_math_expression(expression):
    try:
        # Remove any potential code execution
        if any(keyword in expression for keyword in ["__", "exec", "eval", "import", "open", "os", "sys"]):
            return "Invalid expression: potentially unsafe operations detected."
        
        # Parse and evaluate the expression safely
        result = sympy.sympify(expression)
        return result
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Safe code analysis using AST (no execution)
def analyze_code(code_snippet):
    try:
        # Parse the code without executing it
        parsed = ast.parse(code_snippet)
        
        # Simple code analysis
        imports = []
        functions = []
        classes = []
        
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return {
            "analysis": {
                "imports": imports,
                "functions": functions,
                "classes": classes,
            },
            "message": "Code analyzed successfully. For security reasons, code execution is disabled."
        }
    except SyntaxError as e:
        return f"Syntax error in code: {str(e)}"
    except Exception as e:
        return f"Error analyzing code: {str(e)}"

# Load conversation history for a session
def load_session(session_id):
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        try:
            with open(session_file, "r") as f:
                return json.load(f)
        except:
            return {"history": []}
    return {"history": []}

# Save conversation history for a session
def save_session(session_id, session_data):
    session_file = SESSION_DIR / f"{session_id}.json"
    with open(session_file, "w") as f:
        json.dump(session_data, f)

# Process user message and generate response
def process_message(message, session_id):
    # Load session history
    session = load_session(session_id)
    history = session["history"]
    
    # Add user message to history
    history.append({"role": "user", "message": message, "timestamp": datetime.now().isoformat()})
    
    # Process the message
    response = generate_response(message, history)
    
    # Add bot response to history
    history.append({"role": "bot", "message": response, "timestamp": datetime.now().isoformat()})
    
    # Save updated session
    session["history"] = history
    save_session(session_id, session)
    
    return response

# Generate a response based on user message and conversation history
def generate_response(message, history):
    # Lowercased message for pattern matching
    message_lower = message.lower()
    
    # Greetings
    if any(message_lower.startswith(greeting) for greeting in ["hello", "hi", "hey", "greetings"]):
        return "Hello! How can I assist you with your Python questions today?"
    
    # Farewells
    if any(message_lower.startswith(farewell) for farewell in ["bye", "goodbye", "exit", "quit"]):
        return "Goodbye! Feel free to return if you have more questions."
    
    # Gratitude
    if any(word in message_lower for word in ["thank", "thanks", "appreciate"]):
        return "You're welcome! If you have any more questions, I'm here to help."
    
    # Help request
    if message_lower.startswith("help"):
        return ("I can help you with:\n"
                "1. Answering Python programming questions\n"
                "2. Explaining programming concepts\n"
                "3. Analyzing Python code snippets\n"
                "4. Evaluating mathematical expressions\n"
                "5. Providing information about AI and data science topics")
    
    # Time-based greetings
    if message_lower.startswith("good"):
        for time_period in ["morning", "afternoon", "evening", "night"]:
            if time_period in message_lower:
                return f"Good {time_period}! How can I assist you today?"
        return "Good day! How can I assist you today?"
    
    # Knowledge base queries
    if message_lower.startswith(("what is", "what are", "define", "explain")):
        for keyword, explanation in KNOWLEDGE_BASE.items():
            if keyword in message_lower:
                return explanation
    
    # Code snippet detection (enclosed in ```python ... ```)
    code_pattern = r"```python(.*?)```"
    code_match = re.search(code_pattern, message, re.DOTALL)
    if code_match:
        code_snippet = code_match.group(1).strip()
        analysis_result = analyze_code(code_snippet)
        return f"Code Analysis Result:\n{json.dumps(analysis_result, indent=2)}"
    
    # Math expression detection
    math_pattern = r'\b(\d+(\.\d+)?(\s*[\+\-\*/%\^]\s*\d+(\.\d+)?)+)\b'
    math_match = re.search(math_pattern, message)
    if math_match:
        expression = math_match.group(0)
        result = evaluate_math_expression(expression)
        return f"The result of {expression} is {result}"
    
    # Check for contextual follow-up based on history
    if len(history) >= 2:
        last_bot_message = next((item for item in reversed(history) 
                               if item["role"] == "bot"), None)
        if last_bot_message and "follow-up" in message_lower:
            # Handle follow-up to previous topic
            return "I'll provide more details on our previous topic..."
    
    # If no specific pattern matched, use the transformer model
    # Create a better prompt with context from recent history
    context = ""
    if len(history) > 0:
        # Get the last 3 exchanges for context
        recent_history = history[-min(6, len(history)):]
        context = " ".join([f"{item['role']}: {item['message']}" for item in recent_history])
    
    prompt = f"{context} User: {message}\nBot:"
    output = chatbot(prompt, max_length=1000, truncation=True, pad_token_id=50256, do_sample=True)
    response = output[0]['generated_text'].split("Bot:")[-1].strip()
    
    # If the response is empty or too generic, provide a fallback
    if not response or response == message or len(response) < 5:
        response = "I understand you're asking about something, but I'm not sure how to respond specifically. Could you rephrase or provide more details?"
    
    return response

# Gradio interface with session management
def chat_with_bot(message, state=None):
    # Initialize or retrieve session ID
    if state is None or "session_id" not in state:
        session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        state = {"session_id": session_id}
    else:
        session_id = state["session_id"]
    
    # Process the message
    response = process_message(message, session_id)
    
    return response, state

# Create the Gradio interface with improved UI
demo = gr.Interface(
    fn=chat_with_bot,
    inputs=[
        gr.Textbox(label="Your Message", placeholder="Enter your message here...", lines=3),
        "state"
    ],
    outputs=[
        gr.Textbox(label="Bot Response"),
        "state"
    ],
    title="Enhanced Python Assistant",
    description="A Python chatbot with code analysis, math evaluation, and conversation memory.",
    examples=[
        ["Hello, how can you help me with Python?"],
        ["What is machine learning?"],
        ["Can you analyze this code?\n```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```"],
        ["Calculate 5 + 10 * 2"],
        ["Thank you for your help!"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)