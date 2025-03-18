from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import pytesseract
import tensorflow as tf
from transformers import BartForConditionalGeneration, BartTokenizer
from textblob import TextBlob
import moviepy.editor as mp
from bs4 import BeautifulSoup
import requests
import torch
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import sent_tokenize
from nltk import CFG
from nltk.corpus import stopwords
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')
def generate_random_cfg_tree(original_text):
    try:
        sentences = sent_tokenize(original_text)  # Tokenize into sentences
        
        # Choose a random sentence from the original text
        random_sentence = sentences[0] if sentences else None
        if not random_sentence:
            return "Error: No sentences found in the input text."
        
        # Tokenize the sentence into words
        words = nltk.word_tokenize(random_sentence)
        
        # Define a simple CFG grammar (you can modify this based on your needs)
        grammar = CFG.fromstring("""
            S -> NP VP
            NP -> DT NN | DT NNS | PRP
            VP -> VBD NP | VBZ NP | VBP NP
            DT -> 'The' | 'a' | 'an'
            NN -> 'elephant' | 'dog' | 'cat'
            NNS -> 'elephants' | 'dogs' | 'cats'
            PRP -> 'he' | 'she' | 'it'
            VBD -> 'saw' | 'ate'
            VBZ -> 'sees' | 'eats'
            VBP -> 'see' | 'eat'
        """)
        
        # Create a parser for the CFG grammar
        parser = nltk.ChartParser(grammar)
        
        # Try parsing the sentence and generate a tree
        parse_trees = list(parser.parse(words))
        
        if parse_trees:
            tree = parse_trees[0]  # Take the first parse tree
            tree_str = tree.pformat()  # Convert the tree to a readable format
            return tree_str
        else:
            return "Error generating CFG tree: No valid parse found."
    except Exception as e:
        return f"Error generating tree: {str(e)}"

# Define a function to find emotion-related words in the text
def find_emotion_words(text):
    try:
        # List of emotion-related words (for demonstration, you can expand this list)
        emotion_words = ['happy', 'sad', 'angry', 'joy', 'fear', 'surprised', 'disgust', 'love', 'hate']
        
        # Tokenize the text into words
        words = nltk.word_tokenize(text)
        
        # Filter out stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
        
        # Find emotion words in the filtered list
        emotion_found = [word for word in filtered_words if word in emotion_words]
        
        return emotion_found
    except Exception as e:
        return f"Error finding emotion words: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def summarize_content():
    if request.method == "POST":
        original_text = request.form.get('text')  # Get the original text from the form
        
        # Check if the original text is empty or None
        if not original_text:
            return render_template('error.html', message="No text provided for summarization.")
        
        # Generate CFG tree for a random sentence from the original text
        random_cfg_tree = generate_random_cfg_tree(original_text)
        
        # Find emotion words in the original text
        emotion_words = find_emotion_words(original_text)
        
        # Return results
        return render_template('output.html', result=original_text, emotion_words=emotion_words, random_cfg_tree=random_cfg_tree)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        text_input = request.form.get('text_input')
        url_input = request.form.get('url')
        uploaded_file = request.files.get('file')
        video_file = request.files.get('video')

        # Prepare form data and redirect to the summarize_content route
        if text_input:
            return redirect(url_for('summarize_content', content_type='text', content=text_input))
        elif url_input:
            return redirect(url_for('summarize_content', content_type='url', content=url_input))
        elif uploaded_file and allowed_file(uploaded_file.filename):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)
            return redirect(url_for('summarize_content', content_type='image', content=image_path))
        elif video_file and allowed_file(video_file.filename):
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)
            return redirect(url_for('summarize_content', content_type='video', content=video_path))

    return render_template('main.html')


# Extract text from an image
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)

    # Check if the image is loaded correctly
    if img is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    extracted_text = pytesseract.image_to_string(binary_img)
    return extracted_text

# Extract text from a video (unchanged)
def extract_text_from_video(video_path):
    if not os.path.exists(video_path):
        raise ValueError("Video file does not exist.")
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    video_clip = mp.VideoFileClip(video_path)
    
    if video_clip.audio is None:
        raise ValueError("No audio found in the video.")
    try:
        video_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        raise ValueError(f"Failed to extract audio from video: {e}")

    extracted_text = extract_text_from_audio(audio_path)
    video_clip.close()
    os.remove(audio_path)
    
    return extracted_text

def extract_text_from_audio(audio_path):
    # Placeholder for speech-to-text integration
    return "Extracted speech text from audio (placeholder)."

# Extract text from a URL (scraping)
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    page_text = ' '.join([para.get_text() for para in paragraphs])
    return page_text

MAX_INPUT_LENGTH = 1024  # Max length based on model's constraints

def summarize_text(content):
    inputs = tokenizer(content, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length', return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
