import re
import os
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji


INPUT_FILE = "100_feedback_reviews.xlsx"          # change this to your file's name
OUTPUT_FILE = "lots_of_column_feedback_ranked.xlsx"

# Leave as None to auto-detect the column. Set an exact column name (string)
# to override, e.g. TEXT_COLUMN = "Customer Feedback"
TEXT_COLUMN = None
RATING_COLUMN = None

# Weights used when a row has ALL of rating + text + emoji.
# These auto-rebalance per row depending on what's actually present
# (e.g. if a row has no emoji, its weight shifts to text).
WEIGHT_RATING = 0.50
WEIGHT_TEXT = 0.35
WEIGHT_EMOJI = 0.15

# Tier thresholds on the final 0-1 composite score.
TIERS = [
    (0.80, "Best"),
    (0.60, "Good"),
    (0.40, "Average"),
    (0.20, "Poor"),
    (0.00, "Worst"),
]

# ==================== EMOJI SENTIMENT LEXICON ====================
# Curated scores from -1 (very negative) to +1 (very positive).
# Emojis not in this list are treated as neutral (0.0) but still counted
# as "an emoji was present" for weighting purposes.
EMOJI_LEXICON = {
    "😀": 0.8, "😃": 0.8, "😄": 0.9, "😁": 0.8, "😆": 0.7, "😊": 0.9,
    "🙂": 0.5, "😍": 1.0, "🥰": 1.0, "😘": 0.8, "👍": 0.8, "👏": 0.8,
    "🎉": 0.9, "🔥": 0.6, "💯": 0.9, "❤️": 0.9, "❤": 0.9, "✅": 0.6,
    "⭐": 0.7, "🌟": 0.8, "😎": 0.6, "🤩": 0.9, "😇": 0.7, "🙌": 0.8,
    "💪": 0.6, "😌": 0.4,
    "😐": 0.0, "😑": -0.1, "🤔": 0.0, "😶": 0.0, "🙄": -0.4,
    "😕": -0.4, "😟": -0.5, "😔": -0.5, "😞": -0.6, "😢": -0.6,
    "😭": -0.7, "😡": -0.9, "😠": -0.8, "🤬": -1.0, "👎": -0.8,
    "💔": -0.8, "😤": -0.5, "😩": -0.6, "😫": -0.6, "🤮": -0.9,
    "😒": -0.5, "😨": -0.5, "😰": -0.5, "🥺": -0.2,
}

# ==================== HELPERS ====================
def detect_columns(df):
    """Auto-detect the feedback text column and rating column, unless overridden above."""
    text_col = TEXT_COLUMN
    rating_col = RATING_COLUMN

    if text_col is None:
        object_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        keyworded = [c for c in object_cols if any(
            k in str(c).lower() for k in ["feedback", "review", "comment", "text", "message"]
        )]
        if keyworded:
            text_col = keyworded[0]
        elif object_cols:
            # fall back to the text-like column with the longest average content
            text_col = max(
                object_cols,
                key=lambda c: df[c].dropna().astype(str).str.len().mean() if df[c].notna().any() else 0
            )

    if rating_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        keyworded = [c for c in numeric_cols if any(
            k in str(c).lower() for k in ["rating", "star", "score"]
        )]
        if keyworded:
            rating_col = keyworded[0]
        else:
            for c in numeric_cols:
                vals = df[c].dropna()
                if len(vals) and 0 <= vals.min() and vals.max() <= 10:
                    rating_col = c
                    break

    if text_col is None:
        raise ValueError(
            "Couldn't auto-detect a feedback text column. "
            "Set TEXT_COLUMN at the top of the script to your column's exact name."
        )

    return text_col, rating_col


def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


def emoji_sentiment_score(emojis_found):
    """Average lexicon score of found emojis, normalized 0-1. None if no emojis."""
    if not emojis_found:
        return None
    scores = [EMOJI_LEXICON.get(e, 0.0) for e in emojis_found]
    avg = sum(scores) / len(scores)          # -1 .. 1
    return (avg + 1) / 2                     # 0 .. 1


def text_sentiment_score(analyzer, text):
    """VADER compound score, normalized 0-1. None if no usable text."""
    if not isinstance(text, str) or not text.strip():
        return None
    # Strip emojis before running VADER so they don't skew the text-only score
    # (emojis are scored separately via EMOJI_LEXICON).
    clean_text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    if not clean_text.strip():
        return None
    compound = analyzer.polarity_scores(clean_text)["compound"]  # -1 .. 1
    return (compound + 1) / 2


def normalize_rating(value, observed_max):
    if pd.isna(value):
        return None
    scale = observed_max if observed_max and observed_max > 0 else 5
    return max(0.0, min(1.0, value / scale))


def composite_score(rating_norm, text_norm, emoji_norm):
    """
    Combine whichever signals are present for this row, rebalancing weights
    so they always sum to 1 — a row with no emoji just shifts that weight to
    text; a row with no rating shifts to text (0.70) + emoji (0.30).
    """
    weights = {}
    if rating_norm is not None:
        weights["rating"] = WEIGHT_RATING
    weights["text"] = WEIGHT_TEXT if rating_norm is not None else 0.70
    if emoji_norm is not None:
        weights["emoji"] = WEIGHT_EMOJI if rating_norm is not None else 0.30

    total_weight = sum(weights.values())
    score = 0.0
    if "rating" in weights:
        score += (weights["rating"] / total_weight) * rating_norm
    score += (weights["text"] / total_weight) * (text_norm if text_norm is not None else 0.5)
    if "emoji" in weights:
        score += (weights["emoji"] / total_weight) * emoji_norm

    return round(score, 4)


def tier_for_score(score):
    for threshold, label in TIERS:
        if score >= threshold:
            return label
    return TIERS[-1][1]


# ==================== MAIN PIPELINE ====================
def analyze_and_rank(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_excel(input_path)
    if df.empty:
        print("❌ Input file has no rows.")
        sys.exit(1)

    text_col, rating_col = detect_columns(df)
    print(f"📋 Using text column: '{text_col}'" + (f", rating column: '{rating_col}'" if rating_col else " (no rating column detected)"))

    analyzer = SentimentIntensityAnalyzer()
    observed_max = df[rating_col].max() if rating_col else None

    text_scores, emoji_scores, rating_scores, composites, tiers, emoji_lists = [], [], [], [], [], []

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        emojis_found = extract_emojis(text)

        t_score = text_sentiment_score(analyzer, text)
        e_score = emoji_sentiment_score(emojis_found)
        r_score = normalize_rating(row.get(rating_col), observed_max) if rating_col else None

        c_score = composite_score(r_score, t_score, e_score)

        text_scores.append(round(t_score, 4) if t_score is not None else None)
        emoji_scores.append(round(e_score, 4) if e_score is not None else None)
        rating_scores.append(round(r_score, 4) if r_score is not None else None)
        composites.append(c_score)
        tiers.append(tier_for_score(c_score))
        emoji_lists.append("".join(emojis_found) if emojis_found else "")

    df["Emojis Found"] = emoji_lists
    df["Rating (normalized)"] = rating_scores
    df["Text Sentiment (0-1)"] = text_scores
    df["Emoji Sentiment (0-1)"] = emoji_scores
    df["Composite Score (0-1)"] = composites
    df["Feedback Tier"] = tiers

    df = df.sort_values("Composite Score (0-1)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    df.to_excel(output_path, index=False)

    print(f"\n✅ Ranked {len(df)} feedback entries.")
    print(f"💾 Saved to '{output_path}'")
    print("\nTier breakdown:")
    print(df["Feedback Tier"].value_counts().reindex([t[1] for t in TIERS], fill_value=0).to_string())

    return df


if __name__ == "__main__":
    analyze_and_rank(INPUT_FILE, OUTPUT_FILE)

# ==================== SETUP NOTES ====================
# pip install pandas openpyxl vaderSentiment emoji
#
# 1) Set INPUT_FILE at the top to your actual .xlsx filename/path.
# 2) The script auto-detects your feedback text column and rating column by
#    name (looking for words like "feedback"/"review"/"rating"/"star") and
#    falls back to sensible heuristics if it can't find an obvious match.
#    If it picks the wrong column, just set TEXT_COLUMN / RATING_COLUMN
#    explicitly near the top.
# 3) Output is a new Excel file, sorted best -> worst, with a Rank column
#    and a breakdown of each entry's rating/text/emoji contribution plus
#    its overall Tier (Best / Good / Average / Poor / Worst).