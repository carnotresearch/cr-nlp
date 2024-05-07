from transformers import AutoTokenizer
from transformers import pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
nltk.download('vader_lexicon')

def tokenize_text(text, model_name="bert-base-uncased"):
    """
    Tokenize a given text using the Hugging Face Transformers library.

    Parameters:
    - text (str): The input text to tokenize.
    - model_name (str): The name of the pre-trained model to use for tokenization.
                       Default is "bert-base-uncased".

    Returns:
    - tokens (list): List of tokens obtained by tokenizing the input text.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.tokenize(text)

    return tokens


def analyze_sentiment(sentence: str) -> dict:
    """
    Analyze the sentiment of a given sentence using a pre-trained BERT model.

    This function utilizes the 'sentiment-analysis' pipeline from the Hugging Face
    transformers library, which provides an easy-to-use interface to a BERT model
    pre-trained on sentiment analysis tasks.

    Parameters:
    - sentence (str): The sentence for which to analyze the sentiment.

    Returns:
    - dict: A dictionary containing the label ('POSITIVE' or 'NEGATIVE') and the
            associated confidence score.
    
    Example usage:
    sentiment = analyze_sentiment("I love this product!")
    print(sentiment)  # Output might look like: {'label': 'POSITIVE', 'score': 0.999}
    """

    # Load the sentiment analysis pipeline
    classifier = pipeline('sentiment-analysis')

    # Analyze the sentiment of the sentence
    results = classifier(sentence)

    # Return the first result (the most likely sentiment)
    return results[0]




# Helper function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_words(words):
    """
    Lemmatize a list of words.

    This function takes a list of words, determines the part of speech for each word, 
    and then lemmatizes it (converts it to its base or dictionary form) according 
    to its part of speech. It utilizes the NLTK library's WordNetLemmatizer 
    and the part-of-speech tagging to accurately lemmatize each word.

    Parameters:
    - words: A list of words (strings) that you want to lemmatize.

    Returns:
    - A list of lemmatized words.

    Note: This function requires nltk's WordNetLemmatizer and pos_tag to be imported, 
    along with the wordnet corpus and a function get_wordnet_pos(tag) that converts 
    the part-of-speech tagging conventions between nltk and wordnet.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    
    # Get POS tag for each word
    pos_tagged = pos_tag(words)
    
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmatized_words.append(lemmatized_word)
        
    return lemmatized_words

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def analyze_sentiment_vader(text):
    """
    Analyzes the sentiment of a given text using VADER sentiment analysis.

    Parameters:
    - text: A string containing the text to analyze.

    Returns:
    - A dictionary containing the scores for negative, neutral, positive, and compound sentiments.
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores




import nltk
from nltk.stem.porter import PorterStemmer


def stem_words(words):
    """
    Stems a list of words.

    This function applies the Porter Stemming algorithm to a list of words, 
    reducing each word to its root or stem form. It's particularly useful in 
    natural language processing and search applications where the exact form of 
    a word is less important than its root meaning.

    Parameters:
    - words: A list of words (strings) to be stemmed.

    Returns:
    - A list containing the stemmed version of each input word.

    Example:
    >>> stem_words(["running", "jumps", "easily"])
    ['run', 'jump', 'easili']
    
    Note: This function requires the nltk's PorterStemmer to be imported.
    """
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Stem each word in the list
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words



