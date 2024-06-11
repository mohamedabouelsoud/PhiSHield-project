from flask import Flask, request, render_template, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import string

app = Flask(__name__)

# Load and prepare data
try:
    data = pd.read_csv('spam_d.csv')  # Update the path
    print("CSV file loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    data = pd.DataFrame({'Message': [], 'Category': []})  # Provide a fallback
except Exception as e:
    print(f"Unknown error loading CSV file: {e}")
    data = pd.DataFrame({'Message': [], 'Category': []})  # Provide a fallback

try:
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

    # Train the spam email model
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X_train, y_train)
    print("Email spam model trained successfully.")
except Exception as e:
    print(f"Error preparing data or training the email spam model: {e}")
    clf = None

# Load and prepare phishing URL data
try:
    urls_data = pd.read_csv('urldata.csv')  # Update the path
    print("URL CSV file loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading URL CSV file: {e}")
    urls_data = pd.DataFrame({'url': [], 'label': []})  # Provide a fallback
except Exception as e:
    print(f"Unknown error loading URL CSV file: {e}")
    urls_data = pd.DataFrame({'url': [], 'label': []})  # Provide a fallback

try:
    def makeTokens(f):
        tkns_BySlash = str(f.encode('utf-8')).split('/')
        total_Tokens = []
        for i in tkns_BySlash:
            tokens = str(i).split('-')
            tkns_ByDot = []
            for j in range(0, len(tokens)):
                temp_Tokens = str(tokens[j]).split('.')
                tkns_ByDot = tkns_ByDot + temp_Tokens
            total_Tokens = total_Tokens + tokens + tkns_ByDot
        total_Tokens = list(set(total_Tokens))
        if 'com' in total_Tokens:
            total_Tokens.remove('com')
        return total_Tokens


    y = urls_data["label"]
    url_list = urls_data["url"]
    vectorizer = TfidfVectorizer(tokenizer=makeTokens, token_pattern=None)
    X = vectorizer.fit_transform(url_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    print("URL phishing detection model trained successfully.")
except Exception as e:
    print(f"Error preparing data or training the URL phishing detection model: {e}")
    logit = None


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    try:
        if request.method == 'POST':
            email = request.form['textarea']
            if clf:
                prediction = clf.predict([email])[0]
                result = "Spam" if prediction == 1 else "Not Spam"
                return render_template('email.html', result=result)
            else:
                return render_template('email.html', result="Error: Email spam model not loaded")
        return render_template('email.html')
    except Exception as e:
        print(f"Error in /detect route: {e}")
        return render_template('email.html', result="Error: An unexpected error occurred")


@app.route('/services')
def services():
    return render_template('Services.html')


@app.route('/contact_us')
def contact_us():
    return render_template('ContactUS.html')


@app.route('/url_detection', methods=['GET', 'POST'])
def url_detection():
    try:
        if request.method == 'POST':
            url = request.form['url']
            if logit and vectorizer:
                url_prediction = logit.predict(vectorizer.transform([url]))[0]
                url_result = "Phishing URL" if url_prediction == 'bad' else "Safe URL"
                return render_template('url.html', url_result=url_result)
            else:
                return render_template('url.html', url_result="Error: URL phishing detection model not loaded")
        return render_template('url.html')
    except Exception as e:
        print(f"Error in /url_detection route: {e}")
        return render_template('url.html', url_result="Error: An unexpected error         occurred")


@app.route('/email_detection')
def email_detection():
    return render_template('email.html')


@app.route('/password_generator', methods=['GET', 'POST'])
def password_generator():
    password = ""
    if request.method == 'POST':
        length = 12
        characters = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(random.choice(characters) for i in range(length))
    return render_template('passwordgenerator.html', password=password)


@app.route('/awareness')
def cyber_awareness():
    return render_template('awareness.html')


if __name__ == '__main__':
    app.run(debug=True)

