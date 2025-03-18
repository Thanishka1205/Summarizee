# Flask-Based Text Processing and Summarization

## Overview
This project is a Flask-based web application that provides multiple functionalities for text processing and summarization. It supports text extraction from various sources (text input, images, videos, and URLs) and generates concise summaries using the BART model. Additionally, it includes features such as parsing sentences into a Context-Free Grammar (CFG) tree and identifying emotion-related words.

## Features
- **Text Summarization**: Uses Facebook's `bart-large-cnn` model to summarize text.
- **CFG Tree Generation**: Parses sentences into a tree structure using NLTK's Context-Free Grammar.
- **Emotion Word Detection**: Identifies words associated with emotions.
- **Text Extraction from Images**: Uses `pytesseract` OCR to extract text from images.
- **Text Extraction from Videos**: Extracts audio from videos and processes speech-to-text (placeholder).
- **Text Extraction from URLs**: Scrapes web pages to extract text content.
- **File Upload Support**: Accepts text input, images, and videos for processing.

## Technologies Used
- **Flask**: Web framework for Python.
- **OpenCV**: Image processing.
- **Pytesseract**: Optical character recognition (OCR).
- **Transformers (Hugging Face)**: BART model for summarization.
- **TextBlob**: Basic NLP operations.
- **MoviePy**: Video processing.
- **BeautifulSoup**: Web scraping.
- **NLTK**: Natural Language Processing (NLP).
- **TensorFlow & PyTorch**: Deep learning frameworks.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pip
- Tesseract OCR (Download from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract))
- Required Python packages (install using the command below)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/text-processing-flask.git
   cd text-processing-flask
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up Tesseract OCR path (Windows users):
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
5. Run the Flask application:
   ```bash
   python app.py
   ```
6. Open a browser and go to `http://127.0.0.1:5000/`.

## Usage
1. **Upload text, images, videos, or provide a URL.**
2. **Select processing options:**
   - Summarization
   - Emotion detection
   - CFG tree generation
3. **View results on the output page.**

## Folder Structure
```
static/
    uploads/  # Stores uploaded images and videos
templates/
    index.html  # Main page
    output.html  # Results page
app.py  # Flask application entry point
requirements.txt  # Dependencies
README.md  # Project documentation
```

## Future Enhancements
- Improve speech-to-text integration.
- Expand emotion word detection with a sentiment analysis model.
- Provide API support for external integration.



