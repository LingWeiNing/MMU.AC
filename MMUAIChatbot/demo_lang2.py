import os
import csv
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext, font
from dotenv import load_dotenv
from datetime import datetime
import requests
import threading
import json
from langdetect import detect
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import faiss

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is not set properly.")
if not WEATHER_API_KEY:
    raise ValueError("Weather API Key is not set properly.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8  # Stop listening after a short pause

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and process documents
csv_file_path = "C:/Users/yap63/Desktop/Chatbot/en-FAQ.csv"
pdf_file_path = "C:/Users/yap63/Desktop/Chatbot/Academic_Handbook.pdf"

if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_file_path}")

# Load CSV data into a dictionary
def load_csv_data(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row['Question'].strip().lower()] = row['Answer'].strip()
    return data
csv_data = load_csv_data(csv_file_path)

pdf_loader = PyPDFLoader(file_path=pdf_file_path)
pdf_documents = pdf_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=pdf_documents)

vectorstore = FAISS.from_documents(docs, embeddings_model)

if os.path.exists("faiss_index_react.index"):
    vectorstore.index = faiss.read_index("faiss_index_react.index")
else:
    faiss.write_index(vectorstore.index, "faiss_index_react.index")

retriever = vectorstore.as_retriever()
prompt_template = """
    You are a knowledgeable assistant. Answer the following question using the provided context:

    **Question**: {question}

    **Context**:
    {context}

    **Answer**:
    """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Use a valid model name
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type_kwargs={"prompt": prompt},
    retriever=retriever
)

supported_languages = {
    'en': 'English',
    'zh-cn': 'Chinese',
    'ms': 'Malay',
}

# Global variable to store context
context_history = []

def update_context(command, response):
    context_history.append({"command": command, "response": response})
    if len(context_history) > 10:
        context_history.pop(0)

def get_context():
    return " ".join([item["response"] for item in context_history])

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(command):
    sentiment = analyzer.polarity_scores(command)
    return sentiment['compound']

def respond_gui(response):
    output_text.insert(tk.END, "Bot: " + response + "\n")
    engine.say(response)
    if not engine._inLoop:
        engine.runAndWait()

def get_weather(city="Cyberjaya", language="en"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']

        # English
        if language == "en":
            return f"The current weather in {city} is {weather_description} with a temperature of {temperature}°C."
        # Chinese (Simplified)
        elif language == "zh-cn":
            return f"{city}的当前天气是{weather_description}，温度为{temperature}°C。"
        # Malay
        elif language == "ms":
            return f"Cuaca terkini di {city} ialah {weather_description} dengan suhu {temperature}°C."
        else:
            return f"Weather information in {language} is currently unavailable."

    except requests.RequestException as e:
        return f"Sorry, I couldn't retrieve the weather information due to: {e}"

def listen_for_command_gui():
    command = command_entry.get().strip()
    command_entry.delete(0, tk.END)
    process_command_gui(command)

def open_powerpoint():
    respond_gui("Opening PowerPoint.")
    os.system("start powerpnt")

def open_word():
    respond_gui("Opening Microsoft Word.")
    os.system("start winword")

def open_excel():
    respond_gui("Opening Microsoft Excel.")
    os.system("start excel")

def open_onenote():
    respond_gui("Opening OneNote.")
    os.system("start onenote")

def open_academic_handbook():
    if os.path.exists(pdf_file_path):
        respond_gui("Opening Academic Handbook.")
        os.startfile(pdf_file_path)
    else:
        respond_gui("Sorry, I couldn't find the Academic Handbook.")

def process_command_gui(command):
    if not command:
        return

    def run_command_in_background():
        context = get_context()
        sentiment_score = analyze_sentiment(command)
        command_words = set(command.lower().split())

        try:
            
            command_lower = command.lower()
            # Manually override the language for specific keywords
            if "天气" in command_lower or "时间" in command_lower or "日期" in command_lower:
                detected_language = 'zh-cn'  # Force Chinese for these keywords
            elif "cuaca" in command_lower or "masa" in command_lower or "tarikh" in command_lower:
                detected_language = 'ms'  # Force Malay for the keywords
            else:
                detected_language = detect(command)  # Use language detection for other inputs

            if sentiment_score < -0.5:
                respond_gui("I sense you're upset. How can I assist?")
            
            response = None

            # Load command mappings
            with open('commands.json', 'r', encoding='utf-8') as file:
                command_mappings = json.load(file)
                
            for category, commands in command_mappings.items():
                # Check if the user's command matches any predefined commands
                for key_command in commands:
                    if key_command in command_words:  # Partial matching
                        response = commands[key_command]
                        break
                    
            
            # Select the correct response based on the detected language
            if detected_language == 'zh-cn':  # Chinese
                if "天气" in command_lower:
                    response = get_weather(city="Cyberjaya", language=detected_language)
                elif "时间" in command_lower:
                    response = command_mappings["information"]["时间"].format(time=datetime.now().strftime("%H:%M:%S"))
                elif "日期" in command_lower:
                    response = command_mappings["information"]["日期"].format(date=datetime.now().strftime('%Y-%m-%d'))

            elif detected_language == 'ms':  # Malay
                if "cuaca" in command_lower:
                    response = get_weather(city="Cyberjaya", language=detected_language)
                elif "masa" in command_lower:
                    response = command_mappings["information"]["masa"].format(time=datetime.now().strftime("%H:%M:%S"))
                elif "tarikh" in command_lower:
                    response = command_mappings["information"]["tarikh"].format(date=datetime.now().strftime('%Y-%m-%d'))

            else:  # Default to English
                if "weather" in command_lower:
                    response = get_weather(city="Cyberjaya", language=detected_language)
                elif "time" in command_lower:
                    response = command_mappings["information"]["time"].format(time=datetime.now().strftime("%H:%M:%S"))
                elif "date" in command_lower:
                    response = command_mappings["information"]["date"].format(date=datetime.now().strftime('%Y-%m-%d'))

            if response:
                respond_gui(response)
                
            else:
                if any(keyword in command_lower for keyword in ["open slide", "open powerpoint", "open presentation"]):
                    open_powerpoint()
                elif any(keyword in command_lower for keyword in ["open word", "start word", "open microsoft word"]):
                    open_word()
                elif any(keyword in command_lower for keyword in ["open excel", "start excel", "open spreadsheet"]):
                    open_excel()
                elif any(keyword in command_lower for keyword in ["open onenote", "start onenote", "open notes"]):
                    open_onenote()
                elif command_lower == "open academic handbook":
                    open_academic_handbook()
                elif any(keyword in command_lower for keyword in ["goodbye", "bye"]):
                    respond_gui("Goodbye, have a nice day. Before closing the program, please write you feedback here, thanks.") 
                
                else:
                    # Check CSV data first
                    question_key = command.strip().lower()
                    if question_key in csv_data:
                        text_response = csv_data[question_key]
                    else:
                        response = qa.invoke(f"{context} {command}")
                        text_response = response.get("result", "I couldn't find an answer.")
                    respond_gui(text_response)
                    update_context(command, text_response)

        except Exception as e:
            respond_gui(f"There was an error processing your request: {str(e)}")

    threading.Thread(target=run_command_in_background).start()


def stop_talking():
    engine.stop()

def toggle_recording():
    global is_recording
    if is_recording:
        stop_recording()
    else:
        start_recording()

def start_recording():
    global is_recording
    is_recording = True
    record_button.config(text="Stop", command=toggle_recording)
    record_audio()

def stop_recording():
    global is_recording
    is_recording = False
    record_button.config(text="Record", command=toggle_recording)
    stop_talking()  # Stop text-to-speech if speaking

# Audio input functionality with threading
def record_audio():
    def record():
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Adjust sensitivity
            status_label.config(text="Listening...")
            root.update()
            while is_recording:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # 5 sec to start, 10 sec to finish speaking
                    command = recognizer.recognize_google(audio)
                    respond_gui(f"You said: {command}")
                    process_command_gui(command)
                except sr.UnknownValueError:
                    respond_gui("Sorry, I could not understand the audio.")
                except sr.RequestError:
                    respond_gui("Sorry, there was an error with the speech recognition service. Please try again later")
                    stop_recording()
                except sr.WaitTimeoutError:
                    respond_gui("Recording timed out. Please try again.")
                    stop_recording()
            status_label.config(text="Ready")
    threading.Thread(target=record).start()

def clear_output():
    output_text.delete(1.0, tk.END)

# GUI Setup
root = tk.Tk()
root.title("MMU AI Virtual Assistant")

# Colors
bg_color = "#1e1e1e"  # Dark background color
text_color = "#ffffff"  # White text color
button_bg = "#333333"  # Dark button background
button_fg = "#ffffff"  # White button text

# Main Frame
main_frame = tk.Frame(root, padx=10, pady=10, bg=bg_color)
main_frame.pack(padx=20, pady=20)

# Status Label
status_font = font.Font(family="Helvetica", size=12, weight="bold")
status_label = tk.Label(main_frame, text="Ready", font=status_font, bg=bg_color, fg=text_color)
status_label.pack(pady=5)

# Output Text Box
output_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.NORMAL, height=20, width=60, bg="#2e2e2e", fg=text_color, font=("Helvetica", 12))
output_text.pack(pady=10)

# Command Entry Field
command_entry = tk.Entry(main_frame, width=50, font=("Helvetica", 12), bg="#2e2e2e", fg=text_color, borderwidth=2, relief="solid")
command_entry.pack(pady=5)

# Buttons
button_frame = tk.Frame(main_frame, bg=bg_color)
button_frame.pack(pady=10)

submit_button = tk.Button(button_frame, text="Submit", command=listen_for_command_gui, font=("Helvetica", 12), bg=button_bg, fg=button_fg, relief="flat")
submit_button.pack(side=tk.LEFT, padx=5)

record_button = tk.Button(button_frame, text="Record", command=toggle_recording, font=("Helvetica", 12), bg=button_bg, fg=button_fg, relief="flat")
record_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(button_frame, text="Clear", command=clear_output, font=("Helvetica", 12), bg=button_bg, fg=button_fg, relief="flat")
clear_button.pack(side=tk.LEFT, padx=5)

# Start GUI
root.mainloop()