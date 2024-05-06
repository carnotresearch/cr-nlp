import pytest
from myfunctions import tokenize_text, analyze_sentiment, lemmatize_words, stem_words

# FILEPATH: /C:/Users/aryan/OneDrive/Desktop/cr-nlp/tests/test_myfunctions.py


def test_tokenize_text():
    text = "This is a sample sentence."
    tokens = tokenize_text(text)
    assert len(tokens) == 5
    assert tokens[0] == "this"
    assert tokens[1] == "is"
    assert tokens[2] == "a"
    assert tokens[3] == "sample"
    assert tokens[4] == "sentence"

def test_analyze_sentiment():
    text = "I love this movie!"
    sentiment = analyze_sentiment(text)
    assert sentiment == "positive"

def test_lemmatize_words():
    words = ["running", "jumps", "swimming"]
    lemmas = lemmatize_words(words)
    assert len(lemmas) == 3
    assert lemmas[0] == "run"
    assert lemmas[1] == "jump"
    assert lemmas[2] == "swim"

def test_stem_words():
    words = ["running", "jumps", "swimming"]
    stems = stem_words(words)
    assert len(stems) == 3
    assert stems[0] == "run"
    assert stems[1] == "jump"
    assert stems[2] == "swim"