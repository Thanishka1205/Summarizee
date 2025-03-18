import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


data = [
    ("The Amazon rainforest plays a crucial role in maintaining the global climate.", 1),
    ("Electric vehicles are becoming more popular due to their environmental benefits.", 1),
    ("The study shows a strong correlation between sleep and cognitive function.", 1),
    ("Wildlife conservation is essential for maintaining biodiversity.", 1),
    ("Researchers are exploring renewable energy sources to combat climate change.", 1),
    ("Vaccinations have significantly reduced the prevalence of infectious diseases.", 1),
    ("Organic farming can help reduce soil degradation and promote sustainable agriculture.", 1),
    ("Artificial intelligence is transforming industries by automating complex tasks.", 1),
    ("New technologies in medicine are enabling early diagnosis of diseases.", 1),
    ("Climate change is causing more frequent and severe natural disasters.", 1),
    ("The government is investing in infrastructure to improve transportation networks.", 1),
    ("Remote work has become a widespread trend in the wake of the COVID-19 pandemic.", 1),
    ("Water scarcity is a growing concern in many regions around the world.", 1),
    ("Education systems are adapting to better serve the needs of a digital generation.", 1),
    ("Advances in robotics are leading to more efficient manufacturing processes.", 1),
    ("Public health initiatives have helped to curb the spread of infectious diseases.", 1),
    ("Mental health awareness is becoming a key part of public discourse.", 1),
    ("Renewable energy sources like solar and wind power are seeing increased investment.", 1),
    ("The economy is gradually recovering from the recession caused by the pandemic.", 1),
    ("Conservation efforts are being ramped up to protect endangered species.", 1),
    ("Many companies are now adopting remote work as a permanent option.", 1),
    ("Sustainable building practices are being adopted to reduce environmental impact.", 1),
    ("The healthcare industry is undergoing rapid digital transformation.", 1),
    ("The COVID-19 vaccine rollout has been a critical step toward returning to normal life.", 1),
    ("Cryptocurrencies are becoming more mainstream in global financial markets.", 1),
    ("Pollution is a major environmental issue in urban areas.", 1),
    ("Climate activists are calling for urgent action to address global warming.", 1),
    ("Artificial intelligence is playing a larger role in the healthcare industry.", 1),
    ("Machine learning is helping to analyze large datasets in new ways.", 1),
    ("Biodiversity is critical to the health of ecosystems around the world.", 1),
    ("Data privacy regulations are becoming more stringent worldwide.", 1),
    ("New energy storage technologies are helping to balance renewable energy production.", 1),
    ("Telemedicine is making healthcare more accessible in remote areas.", 1),
    ("The rise of e-commerce has drastically changed the retail landscape.", 1),
    ("Environmental regulations are becoming stricter to combat pollution.", 1),
    ("Innovations in biotechnology are improving treatments for genetic disorders.", 1),
    ("Electric vehicle adoption is accelerating as infrastructure improves.", 1),
    ("Food security is a growing concern due to climate change.", 1),
    ("Remote learning has become a critical tool during the pandemic.", 1),
    ("Governments are investing heavily in renewable energy technologies.", 1),
    ("Cybersecurity threats are becoming more frequent in a connected world.", 1),
    ("Genetic engineering is creating new possibilities in agriculture.", 1),
    ("Artificial intelligence is improving the accuracy of medical diagnoses.", 1),
    ("The renewable energy sector is expected to grow exponentially in the coming years.", 1),
    ("Mental health support is becoming more accessible through telemedicine.", 1),
    ("Automation is reducing the need for manual labor in many industries.", 1),
    ("Space exploration is entering a new era with private companies leading the charge.", 1),
    ("Solar energy is one of the fastest-growing sources of renewable energy.", 1),
    ("Urbanization is leading to increased demand for sustainable development.", 1),
    ("The global population is expected to reach 9 billion by 2050.", 1),
    
    # Irrelevant Sentences
    ("Buy one, get one free on all shoes in our store!", 0),
    ("Click here to claim your free trial of this amazing software.", 0),
    ("This site uses cookies to improve your browsing experience.", 0),
    ("Sign up for our newsletter to receive the latest updates.", 0),
    ("Congratulations, you've won a free vacation!", 0),
    ("This is a sponsored post about the best gadgets of 2023.", 0),
    ("Don't miss out on this limited-time offer!", 0),
    ("Your session is about to expire, please log in again.", 0),
    ("Advertise with us to reach millions of potential customers.", 0),
    ("Join our rewards program and start earning points today.", 0),
    ("Check out the latest deals on electronics at our store.", 0),
    ("This website requires JavaScript to function properly.", 0),
    ("Please accept cookies to continue using this site.", 0),
    ("This content is sponsored by our trusted partner.", 0),
    ("Upgrade to premium to access exclusive content.", 0),
    ("These are the top 10 travel destinations for 2023.", 0),
    ("Shop now and save big on your next purchase!", 0),
    ("Subscribe to our YouTube channel for more updates.", 0),
    ("This post contains affiliate links to products we love.", 0),
    ("Join us for a free webinar on digital marketing.", 0),
    ("This page uses cookies to improve performance.", 0),
    ("Limited-time offer: Get 50% off your first purchase.", 0),
    ("Sign in to access personalized content.", 0),
    ("Thank you for visiting our website, come back soon!", 0),
    ("Our website uses cookies for analytics and advertising.", 0),
    ("This content is part of a paid promotion.", 0),
    ("Click here to read more sponsored content.", 0),
    ("Sign up to get exclusive discounts and offers.", 0),
    ("Your order has been confirmed, thank you for shopping!", 0),
    ("Get a free consultation with our experts today.", 0),
    ("You are being redirected, please wait...", 0),
    ("This page is powered by Google Ads.", 0),
    ("Please enable JavaScript to continue.", 0),
    ("Buy now and get free shipping on your order.", 0),
    ("Follow us on social media for the latest news.", 0),
    ("This content is available to premium members only.", 0),
    ("Our store is having a 50% off sale this weekend!", 0),
    ("Upgrade to a premium plan to remove ads.", 0),
    ("This page is not available in your region.", 0),
    ("Sign up today to receive a free gift with your purchase.", 0),
    ("This site is protected by reCAPTCHA.", 0),
    ("Flash sale! Get 40% off all products.", 0),
    ("Earn rewards with every purchase you make.", 0),
    ("Get a free quote from our experts today.", 0),
    ("Thank you for subscribing to our newsletter.", 0),
    ("This offer is valid for a limited time only.", 0),
    ("Enjoy ad-free browsing with our premium membership.", 0),
    ("This is a pop-up ad to inform you about our sale.", 0),
    ("Please verify your email to continue.", 0),
    ("We respect your privacy and use cookies responsibly.", 0)
]

# Separate the sentences and labels
sentences, labels = zip(*data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Build a model pipeline using TF-IDF Vectorizer and Naive Bayes Classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the model (optional step)
def load_model():
    return model

# Function to classify a sentence
def classify_sentence(text):
    return model.predict([text])[0]