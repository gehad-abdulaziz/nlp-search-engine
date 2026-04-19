from src.preprocessing import preprocess_text

def test_lowercase():
    text = "HELLO WORLD"
    result = preprocess_text(text)
    assert "hello" in result
    assert "world" in result

def test_stopwords_removed():
    text = "this is a simple test"
    result = preprocess_text(text)
    assert "is" not in result.split()
    assert "this" not in result.split()

def test_punctuation_removed():
    text = "hello!!!"
    result = preprocess_text(text)
    assert result == "hello"

def test_numbers_removed():
    text = "version 2 has 3 updates"
    result = preprocess_text(text)
    assert "2" not in result.split()
    assert "3" not in result.split()

def test_lemmatization():
    text = "cars"
    result = preprocess_text(text)
    assert "car" in result.split()

def test_empty_input():
    result = preprocess_text(123)
    assert result == ""
