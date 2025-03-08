import gradio as gr
from transformers import pipeline
import math as m
import regex as re

chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def evaluate_code(code):
    if code.startswith("import"):
        return "Import statements are not allowed."

    try:
        # Execute the code and capture the output
        result = eval(code)
    except Exception as e:
        # Capture any exceptions that occur during execution
        result = str(e)

    return result

def chat_with_bot(prompt):
    # Create a conversation with the chatbot
    prompt = prompt.replace("pi", "3.14")  # Example of a simple code transformation
    match = re.search(r'\(*\d+(\.\d+)?(\s*[\+\-\*/%]\s*\(*\d+(\.\d+)?\)*|\s*\*\*\s*\(*\d+(\.\d+)?\)*)+', prompt)

    response = None

    if prompt.lower().startswith("hello"):
        response = "Hello! How can I assist you today?"

    elif prompt.lower().startswith("hi") or prompt.lower().startswith("hey"):
        response = "Hi there! How can I help you?"

    elif prompt.lower().startswith("what's up") or prompt.lower().startswith("how are you"):
        response = "I'm just a bot, but I'm here to help you with your Python code!"

    elif prompt.lower().startswith("thank you") or prompt.lower().startswith("thanks"):
        response = "You're welcome! If you have any more questions, feel free to ask."

    elif prompt.lower().startswith("bye") or prompt.lower().startswith("exit"):
        response = "Goodbye! Have a great day!"

    elif prompt.lower().startswith("help"):
        response = "I can help you with Python code evaluation. Just enter your code and I'll run it for you."

    elif prompt.lower().startswith("good"):
        if "morning" in prompt.lower():
            response = "Good morning! How can I assist you today?"
        elif "afternoon" in prompt.lower():
            response = "Good afternoon! How can I assist you today?"
        elif "evening" in prompt.lower():
            response = "Good evening! How can I assist you today?"
        else:
            response = "Good day! How can I assist you today?"

    elif prompt.lower().startswith("what is"):
        if "ai" in prompt.lower():
            response = "AI stands for Artificial Intelligence. It's the simulation of human intelligence processes by machines."
        elif "python" in prompt.lower():
            response = "Python is a high-level, interpreted programming language known for its readability and versatility."
        elif "gradio" in prompt.lower():
            response = "Gradio is a Python library that allows you to quickly create user interfaces for machine learning models."
        elif "regex" in prompt.lower():
            response = "Regex, or Regular Expressions, is a sequence of characters that form a search pattern. It's used for string matching."
        elif "machine learning" in prompt.lower():
            response = "Machine learning is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions based on data."
        elif "deep learning" in prompt.lower():
            response = "Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data."
        elif "data science" in prompt.lower():
            response = "Data science is a field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data."

    elif match:
        print(match.group())
        response = f"The result is {evaluate_code(match.group())}"

    else:
        output = chatbot(prompt, max_length=1000, truncation=True, pad_token_id=50256, do_sample=True)
        response = output[0]['generated_text'][len(prompt):].strip()

    return response

demo = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(label="Output"),
    title="Python Simple Chatbot",
    description="A simple Chatbot.",
)

if __name__ == "__main__":
    demo.launch(share=True)