import requests
from bs4 import BeautifulSoup
from ml_model import load_model, classify_sentence  
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
model = load_model()

def extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def filter_content(text):
    sentences = text.split('.')
    
    filtered_sentences = [sentence for sentence in sentences if classify_sentence(sentence) == 1]
    
    purified_content = '. '.join(filtered_sentences)
    return purified_content
def main(url):
    content = extract_content(url)
    purified_content = filter_content(content)
    print("Purified Content:\n", purified_content)
input_url = 'https://en.wikipedia.org/wiki/Rohit_Sharma'  
main(input_url)

