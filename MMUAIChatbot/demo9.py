import os
import csv
import pyttsx3
import customtkinter as ctk
from dotenv import load_dotenv
from datetime import datetime
import requests
import threading
import json
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
import faiss
import langid  # Using langid for better language detection

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

if not WEATHER_API_KEY:
    raise ValueError("Weather API Key is not set properly.")

# Initialize text-to-speech engine
engine = pyttsx3.init()
analyzer = SentimentIntensityAnalyzer()
context_history = []

# Initialize embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load and process documents
csv_loader = CSVLoader(file_path="en-FAQ.csv")

documents = csv_loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embeddings_model)

if os.path.exists("faiss_index_react.index"):
    try:
        vectorstore.index = faiss.read_index("faiss_index_react.index")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
else:
    try:
        vectorstore.index = faiss.IndexFlatL2(512)  # Assuming embedding dimension size of 512
        faiss.write_index(vectorstore.index, "faiss_index_react.index")
    except Exception as e:
        print(f"Error creating and saving FAISS index: {e}")

retriever = vectorstore.as_retriever()

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define supported languages
supported_languages = {
    'en': 'English',
    'zh-cn': 'Chinese',
    'ms': 'Malay',
}

def update_context(command, response):
    context_history.append({"command": command, "response": response})
    if len(context_history) > 10:
        context_history.pop(0)

def get_context():
    return " ".join([item["response"] for item in context_history])

def analyze_sentiment(command):
    sentiment = analyzer.polarity_scores(command)
    return sentiment['compound']

def batch_update_gui():
    buffer = []
    def update(response):
        buffer.append(response)
        if len(buffer) >= 10:  # Or some other batch size
            output_text.configure(state="normal")
            output_text.insert(ctk.END, "\n".join(buffer))
            output_text.configure(state="disabled")
            output_text.see(ctk.END)
            buffer.clear()

# Display response in the text box and scroll to the end
def respond_gui(response, lang='en'):
    output_text.configure(state="normal")
    output_text.insert(ctk.END, "Bot: " + response + "\n")
    output_text.configure(state="disabled")
    output_text.see(ctk.END)

    voices = engine.getProperty('voices')
    if lang == 'zh-cn':
        for voice in voices:
            if 'zh' in voice.id:  # Look for Chinese voice
                engine.setProperty('voice', voice.id)
                break
    elif lang == 'ms':
        for voice in voices:
            if 'ms' in voice.id:  # Look for Malay voice (may need more specific checks)
                engine.setProperty('voice', voice.id)
                break
    else:
        engine.setProperty('voice', voices[0].id)  # Default to the first voice (usually English)

    engine.say(response)
    if not engine._inLoop:
        engine.runAndWait()

def get_weather(city="Cyberjaya", lang="en"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        
        # Ensure language is detected and the correct output is returned based on lang
        if lang.lower() == 'zh-cn':
            return f"{city}的天气是{weather_description}，温度是{temperature}°C."
        elif lang.lower() == 'ms':
            return f"Cuaca di {city} adalah {weather_description} dengan suhu {temperature}°C."
        else:
            return f"The current weather in {city} is {weather_description} with a temperature of {temperature}°C."
    
    except requests.RequestException as e:
        return f"Sorry, I couldn't retrieve the weather information due to: {e}"

def listen_for_command_gui():
    command = command_entry.get().strip()
    command_entry.delete(0, ctk.END)
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


def open_academic_handbook():
    pdf_file_path = "Academic_Handbook.pdf"  # Use the file path directly
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
        command_lower = command.lower()

        # Attempt to manually fix language detection for known Malay phrases
        if command_lower in ["hai", "waktu", "siapa awak", "tarikh", "cuaca"]:
            lang = 'ms'
        elif command_lower in ["你好", "你是谁", "时间", "日期", "天气"]:
            lang = 'zh-cn'
        else:
            lang, _ = langid.classify(command)

        print(f"Detected language: {lang} for command: {command}")  # Debugging output

        if lang not in supported_languages:
            app.after(0, lambda: respond_gui("Sorry, I cannot process that language.", lang='en'))
            return

        # Detect negative sentiment
        if sentiment_score < -0.7:
            if lang == 'zh-cn':
                app.after(0, lambda: respond_gui("我感觉你很不高兴。我该如何帮助你？", lang='zh-cn'))
            elif lang == 'ms':
                app.after(0, lambda: respond_gui("Saya rasa awak kecewa. Bagaimana boleh saya bantu?", lang='ms'))
            else:
                app.after(0, lambda: respond_gui("I sense you're upset. How can I assist?"))
            return

        try:
            with open('commands.json', 'r', encoding='utf-8') as file:
                command_mappings = json.load(file)
        except FileNotFoundError:
            app.after(0, lambda: respond_gui("Command configuration file not found."))
            command_mappings = {}  # Set to empty or provide defaults
        except json.JSONDecodeError:
            app.after(0, lambda: respond_gui("Error parsing command configuration."))
            command_mappings = {}  # Set to empty or provide defaults

        # Adjust language detection logic to treat 'zh' as 'zh-cn'
        if lang == 'zh-cn':
            print("Processing Chinese command...")
            if "你好" in command:
                app.after(0, lambda: respond_gui("你好，有什么我可以帮您的吗？", lang='zh-cn'))
                return
            elif "你是谁" in command:
                app.after(0, lambda: respond_gui("你好！我是一个名为 Alex 的人工智能聊天机器人，我可以帮助回答您的问题。", lang='zh-cn'))
                return
            elif "时间" in command:
                current_time = get_current_time()
                app.after(0, lambda: respond_gui(f"现在的时间是 {current_time}.", lang='zh-cn'))
                return
            elif "日期" in command:
                current_date = get_current_date()
                app.after(0, lambda: respond_gui(f"现在的日期是 {current_date}.", lang='zh-cn'))
                return
            elif "天气" in command:
                weather_info = get_weather(city="Cyberjaya", lang="zh-cn")
                app.after(0, lambda: respond_gui(weather_info, lang='zh-cn'))
                return

        # Check for Malay commands
        elif lang.startswith('ms'):
            lang = 'ms'  # Normalize Malay
            print("Processing Malay command...")
            if "hai" in command_lower:
                app.after(0, lambda: respond_gui("Hai! Bagaimana saya boleh membantu hari ini?", lang='ms'))
                return
            elif "siapa awak" in command_lower:
                app.after(0, lambda: respond_gui("Hello! Saya adalah AI Chatbot bernama Alex.", lang='ms'))
                return
            elif "waktu" in command_lower:
                current_time = get_current_time()
                app.after(0, lambda: respond_gui(f"Sekarang waktu: {current_time}.", lang='ms'))
                return
            elif "tarikh" in command_lower:
                current_date = get_current_date()
                app.after(0, lambda: respond_gui(f"Tarikh hari ini ialah {current_date}.", lang='ms'))
                return
            elif "cuaca" in command_lower:
                weather_info = get_weather(city="Cyberjaya", lang="ms")
                app.after(0, lambda: respond_gui(weather_info, lang='ms'))
                return

        # Handle English commands
        else:
            if any(greeting in command_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
                app.after(0, lambda: respond_gui("Hello! How can I help you today?"))
            elif any(who in command_lower for who in ["who are you", "what are you", "can you tell me who you are"]):
                app.after(0, lambda: respond_gui("Hello! I am an AI Chatbot named Alex, I can help by answering your questions."))
            elif "time" in command_lower:
                current_time = get_current_time()
                app.after(0, lambda: respond_gui(f"The current time is {current_time}."))
            elif "date" in command_lower:
                current_date = get_current_date()
                app.after(0, lambda: respond_gui(f"The current date is {current_date}."))
            elif "weather" in command_lower:
                weather_info = get_weather(city="Cyberjaya", lang="en")
                app.after(0, lambda: respond_gui(weather_info))
            elif command_lower == "stop":
                stop_talking()

            elif any(keyword in command_lower for keyword in ["open slides", "open powerpoint", "open presentation"]):
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
                respond_gui("Goodbye, have a nice day.")
             
            else:
                context_to_query = f"Context: {context}. Query: {command}"
                response = qa.invoke(context_to_query)
                text_response = response.get("result", "I couldn't find an answer.")
                app.after(0, lambda: respond_gui(text_response))
                update_context(command, text_response)

    threading.Thread(target=run_command_in_background).start()

def stop_talking():
    engine.stop()

def listen_for_audio_command():
    threading.Thread(target=_listen_in_background).start()

def _listen_in_background():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        transcription = recognizer.recognize_google(audio)
        print(f"You said: {transcription}")
        app.after(0, lambda: process_command_gui(transcription))  # Update GUI safely
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestException as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_current_date():
    now = datetime.now()
    return now.strftime("%Y-%m-%d")

# Feedback Collection
feedback_file_path = "Feedback.csv"

# Check if the feedback file exists, if not create it
if not os.path.isfile(feedback_file_path):
    with open(feedback_file_path, mode='w', newline='', encoding='utf-8') as feedback_file:
        feedback_writer = csv.writer(feedback_file)
        feedback_writer.writerow(['Timestamp', 'Feedback'])

# Collect feedback
def collect_feedback():
    feedback = output_text.get("1.0", ctk.END).strip()
    if feedback:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(feedback_file_path, mode='a', newline='', encoding='utf-8') as feedback_file:
            feedback_writer = csv.writer(feedback_file)
            feedback_writer.writerow([timestamp, feedback])
        print("Feedback successfully saved.")
    else:
        print("No feedback provided.")

# GUI Setup
ctk.set_appearance_mode("Dark")  # Dark mode
ctk.set_default_color_theme("blue")  # Change the theme color to blue

app = ctk.CTk()
app.geometry("600x400")  # Increased size for better user experience
app.title("MMU Innov8 Lab AI Assistant")

# Command Entry
command_entry = ctk.CTkEntry(app, width=500, placeholder_text="Type your command here...")
command_entry.pack(pady=20)

# Submit Button
submit_button = ctk.CTkButton(app, text="Submit", command=listen_for_command_gui, width=150)
submit_button.pack(pady=10)

# Output Text Box
output_text = ctk.CTkTextbox(app, width=580, height=200)
output_text.pack(pady=10)
output_text.configure(state="disabled")  # Start as disabled

# Audio Command Button
audio_button = ctk.CTkButton(app, text="Speak", command=listen_for_audio_command, width=150)
audio_button.pack(pady=10)

# Run the GUI
app.mainloop()