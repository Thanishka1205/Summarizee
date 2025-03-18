import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the T5 model and tokenizer on the appropriate device
model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-large")

def summarize_chunk(text, max_length=150, min_length=30):
    """Summarizes a single chunk of text."""
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=8,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def sliding_window_summarization(text, window_size=450, overlap=50):
    """Applies sliding window summarization to handle long texts."""
    words = text.split()
    summaries = []

    # Process the text in overlapping windows
    for i in range(0, len(words), window_size - overlap):
        window = ' '.join(words[i:i + window_size])
        summary = summarize_chunk(window)
        summaries.append(summary)

    # Combine all summaries into one final summary
    return ' '.join(summaries)

# Get user input
user_text = input("Enter the text you want to summarize:\n")

# Display the final summary
print("\nSummary:\n", sliding_window_summarization(user_text))  