"""
sentiment_analysis.py
---------------------
VADER-based sentiment analysis on cleaned review text.
Falls back to a simple lexicon approach when NLTK data is unavailable.
"""

import re
import os

# ── Try importing NLTK VADER ──────────────────────────────────────────────────
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _SIA = SentimentIntensityAnalyzer()
    _USE_VADER = True
except Exception:
    _USE_VADER = False


# ── Simple fallback lexicon ───────────────────────────────────────────────────
_POS_WORDS = {"love","great","excellent","amazing","fantastic","superb",
              "outstanding","perfect","happy","satisfied","wonderful","best",
              "awesome","good","nice","recommend","quality","fast","pleased"}
_NEG_WORDS = {"terrible","horrible","worst","awful","poor","bad","hate",
              "disappointed","broken","waste","scam","junk","useless",
              "damaged","failed","avoid","disgust","defective","cheap"}


def _fallback_sentiment(text: str) -> str:
    words = set(str(text).lower().split())
    pos   = len(words & _POS_WORDS)
    neg   = len(words & _NEG_WORDS)
    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"


# ── Public API ────────────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> dict:
    """
    Returns a dict with keys:
      label    : 'Positive' | 'Neutral' | 'Negative'
      compound : float in [-1, 1]  (VADER) or heuristic score
    """
    cleaned = re.sub(r"[^a-z\s]", " ", str(text).lower()).strip()

    if _USE_VADER:
        scores   = _SIA.polarity_scores(cleaned)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return {"label": label, "compound": compound, "method": "VADER"}

    label = _fallback_sentiment(cleaned)
    score_map = {"Positive": 0.6, "Neutral": 0.0, "Negative": -0.6}
    return {"label": label, "compound": score_map[label], "method": "lexicon"}


def annotate_dataframe(df, review_col: str = "cleaned_review"):
    """Add sentiment_pred and compound columns to a DataFrame in-place."""
    results  = df[review_col].apply(predict_sentiment)
    df["sentiment_pred"] = results.apply(lambda r: r["label"])
    df["compound"]       = results.apply(lambda r: r["compound"])
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Absolutely love this product! Works perfectly and ships fast.",
        "Okay product, nothing special but does the job.",
        "Terrible! Broke after two days. Complete waste of money.",
    ]
    print(f"Using: {'VADER' if _USE_VADER else 'fallback lexicon'}\n")
    for s in samples:
        r = predict_sentiment(s)
        print(f"  [{r['label']:8s}  {r['compound']:+.3f}]  {s[:60]}")
