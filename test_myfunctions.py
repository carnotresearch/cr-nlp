import pytest
from myfunctions import tokenize_text, analyze_sentiment, lemmatize_words, stem_words

# FILEPATH: /C:/Users/aryan/OneDrive/Desktop/cr-nlp/tests/test_myfunctions.py






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