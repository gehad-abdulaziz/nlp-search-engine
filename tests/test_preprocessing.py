from src.preprocessing import preprocess_text


def test_lowercase():
    text = "HELLO WORLD"
    tokens = preprocess_text(text)
    assert "hello" in tokens
    assert "world" in tokens


def test_stopwords_removed():
    text = "this is a simple test"
    tokens = preprocess_text(text)
    assert "is" not in tokens
    assert "this" not in tokens


def test_punctuation_removed():
    text = "hello!!!"
    tokens = preprocess_text(text)
    assert tokens == ["hello"]


def test_numbers_removed():
    text = "version 2 has 3 updates"
    tokens = preprocess_text(text)
    assert "2" not in tokens
    assert "3" not in tokens


def test_lemmatization():
    text = "cars"
    tokens = preprocess_text(text)
    assert "car" in tokens