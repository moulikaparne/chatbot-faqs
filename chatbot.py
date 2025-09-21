import tkinter as tk
from tkinter import scrolledtext
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Preprocessing function
# -------------------------
def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text

# -------------------------
# FAQs (Q&A pairs)
# -------------------------
questions = [
    "What is your name?",
    "How does this chatbot work?",
    "What is Python?",
    "How to install Python?",
    "What can you do?"
]

answers = [
    "I am your FAQ Chatbot 🤖",
    "I work by matching your question with the most similar FAQ using cosine similarity.",
    "Python is a popular programming language for AI, ML, and web development.",
    "You can download and install Python from the official website: https://python.org",
    "I can answer FAQs that are predefined in my system!"
]

# -------------------------
# Vectorize questions
# -------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([preprocess(q) for q in questions])

def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    return answers[index]

# -------------------------
# Tkinter GUI
# -------------------------
def send():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_area.insert(tk.END, "You: " + user_input + "\n", "user")
    
    if user_input.lower() == "exit":
        chat_area.insert(tk.END, "Chatbot: Goodbye! 👋\n", "bot")
        root.after(1000, root.quit)  # close after 1 second
        return
    
    response = chatbot_response(user_input)
    chat_area.insert(tk.END, "Chatbot: " + response + "\n\n", "bot")
    entry.delete(0, tk.END)

# Exit button function
def close_chat():
    chat_area.insert(tk.END, "Chatbot: Goodbye! 👋\n", "bot")
    root.after(500, root.quit)

# Main window
root = tk.Tk()
root.title("FAQ Chatbot 🤖")
root.geometry("500x580")
root.config(bg="#f2f2f2")

# Chat area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 12), bg="white", fg="black")
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Style tags
chat_area.tag_config("user", foreground="blue")
chat_area.tag_config("bot", foreground="green")

# Entry + Button frame
frame = tk.Frame(root, bg="#f2f2f2")
frame.pack(padx=10, pady=10, fill=tk.X)

entry = tk.Entry(frame, font=("Arial", 14))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

send_button = tk.Button(frame, text="Send", command=send, font=("Arial", 12), bg="#4CAF50", fg="white")
send_button.pack(side=tk.LEFT, padx=(0, 10))

exit_button = tk.Button(frame, text="Exit", command=close_chat, font=("Arial", 12), bg="#f44336", fg="white")
exit_button.pack(side=tk.RIGHT)

# Initial message
chat_area.insert(tk.END, "Chatbot: Hello! Ask me anything (type 'exit' or press Exit button to quit).\n\n", "bot")

root.mainloop()