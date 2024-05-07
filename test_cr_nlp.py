import pytest
from cr_nlp import tokenize_text, analyze_sentiment, lemmatize_words, analyze_sentiment_vader, stem_words

# Assuming your_module_name is the name of the Python file containing your code.

def test_tokenize_text():
    text = "Hello, world! This is a test sentence."
    tokens = tokenize_text(text)
    assert isinstance(tokens, list)
    assert "hello" in tokens  # checking tokenization lowercasing in 'bert-base-uncased'

def test_analyze_sentiment():
    sentence = "I love this product!"
    result = analyze_sentiment(sentence)
    assert isinstance(result, dict)
    assert 'label' in result
    assert result['label'] in ['POSITIVE', 'NEGATIVE']

def test_lemmatize_words():
    words = ['running', 'jumps', 'easily']
    lemmatized_words = lemmatize_words(words)
    assert lemmatized_words == ['run', 'jump', 'easily']  # Expected lemmatized forms

def test_analyze_sentiment_vader():
    text = "I really hate this weather!"
    result = analyze_sentiment_vader(text)
    assert isinstance(result, dict)
    assert 'neg' in result and result['neg'] > 0  # Expecting a negative sentiment

def test_stem_words():
    words = ["running", "jumps", "easily"]
    stemmed_words = stem_words(words)
    assert stemmed_words == ['run', 'jump', 'easili']  # Expected stemmed forms


