import re
import io
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Feedback Ranker",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHT_RATING = 0.50
WEIGHT_TEXT = 0.35
WEIGHT_EMOJI = 0.15

TIERS = [
    (0.80, "Best"),
    (0.60, "Good"),
    (0.40, "Average"),
    (0.20, "Poor"),
    (0.00, "Worst"),
]


# ============================================================
# EMOJI SENTIMENT LEXICON
# ============================================================

EMOJI_LEXICON = {
    "😀": 0.8, "😃": 0.8, "😄": 0.9, "😁": 0.8,
    "😆": 0.7, "😊": 0.9, "🙂": 0.5, "😍": 1.0,
    "🥰": 1.0, "😘": 0.8, "👍": 0.8, "👏": 0.8,
    "🎉": 0.9, "🔥": 0.6, "💯": 0.9, "❤️": 0.9,
    "❤": 0.9, "✅": 0.6, "⭐": 0.7, "🌟": 0.8,
    "😎": 0.6, "🤩": 0.9, "😇": 0.7, "🙌": 0.8,
    "💪": 0.6, "😌": 0.4,

    "😐": 0.0, "😑": -0.1, "🤔": 0.0, "😶": 0.0,
    "🙄": -0.4, "😕": -0.4, "😟": -0.5, "😔": -0.5,
    "😞": -0.6, "😢": -0.6, "😭": -0.7, "😡": -0.9,
    "😠": -0.8, "🤬": -1.0, "👎": -0.8, "💔": -0.8,
    "😤": -0.5, "😩": -0.6, "😫": -0.6, "🤮": -0.9,
    "😒": -0.5, "😨": -0.5, "😰": -0.5, "🥺": -0.2,
}


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(df):

    text_col = None
    rating_col = None

    # ----------------------------
    # Detect text column
    # ----------------------------

    object_cols = [
        c for c in df.columns
        if pd.api.types.is_string_dtype(df[c])
        or df[c].dtype == object
    ]

    keyworded = [
        c for c in object_cols
        if any(
            k in str(c).lower()
            for k in [
                "feedback",
                "review",
                "comment",
                "text",
                "message"
            ]
        )
    ]

    if keyworded:
        text_col = keyworded[0]

    elif object_cols:

        text_col = max(
            object_cols,
            key=lambda c:
            df[c].dropna().astype(str).str.len().mean()
            if df[c].notna().any()
            else 0
        )

    # ----------------------------
    # Detect rating column
    # ----------------------------

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    keyworded = [
        c for c in numeric_cols
        if any(
            k in str(c).lower()
            for k in [
                "rating",
                "star",
                "score"
            ]
        )
    ]

    if keyworded:

        rating_col = keyworded[0]

    else:

        for c in numeric_cols:

            vals = df[c].dropna()

            if (
                len(vals)
                and 0 <= vals.min()
                and vals.max() <= 10
            ):
                rating_col = c
                break

    if text_col is None:

        raise ValueError(
            "Could not automatically detect the feedback text column."
        )

    return text_col, rating_col


# ============================================================
# EMOJI FUNCTIONS
# ============================================================

def extract_emojis(text):

    if not isinstance(text, str):
        return []

    return [
        e["emoji"]
        for e in emoji.emoji_list(text)
    ]


def emoji_sentiment_score(emojis_found):

    if not emojis_found:
        return None

    scores = [
        EMOJI_LEXICON.get(e, 0.0)
        for e in emojis_found
    ]

    avg = sum(scores) / len(scores)

    return (avg + 1) / 2


# ============================================================
# TEXT SENTIMENT
# ============================================================

def text_sentiment_score(analyzer, text):

    if not isinstance(text, str):
        return None

    if not text.strip():
        return None

    # Remove emojis before VADER analysis
    clean_text = re.sub(
        r'[^\w\s.,!?\'"-]',
        '',
        text
    )

    if not clean_text.strip():
        return None

    compound = analyzer.polarity_scores(
        clean_text
    )["compound"]

    return (compound + 1) / 2


# ============================================================
# RATING NORMALIZATION
# ============================================================

def normalize_rating(value, observed_max):

    if pd.isna(value):
        return None

    scale = (
        observed_max
        if observed_max and observed_max > 0
        else 5
    )

    return max(
        0.0,
        min(1.0, value / scale)
    )


# ============================================================
# COMPOSITE SCORE
# ============================================================

def composite_score(
    rating_norm,
    text_norm,
    emoji_norm
):

    weights = {}

    if rating_norm is not None:
        weights["rating"] = WEIGHT_RATING

    if rating_norm is not None:
        weights["text"] = WEIGHT_TEXT
    else:
        weights["text"] = 0.70

    if emoji_norm is not None:

        if rating_norm is not None:
            weights["emoji"] = WEIGHT_EMOJI
        else:
            weights["emoji"] = 0.30

    total_weight = sum(
        weights.values()
    )

    score = 0.0

    if "rating" in weights:

        score += (
            weights["rating"]
            / total_weight
        ) * rating_norm

    score += (
        weights["text"]
        / total_weight
    ) * (
        text_norm
        if text_norm is not None
        else 0.5
    )

    if "emoji" in weights:

        score += (
            weights["emoji"]
            / total_weight
        ) * emoji_norm

    return round(score, 4)


# ============================================================
# TIER
# ============================================================

def tier_for_score(score):

    for threshold, label in TIERS:

        if score >= threshold:
            return label

    return "Worst"


# ============================================================
# ANALYZE DATA
# ============================================================

def analyze_feedback(df):

    if df.empty:

        raise ValueError(
            "The uploaded Excel file contains no data."
        )

    text_col, rating_col = detect_columns(df)

    analyzer = SentimentIntensityAnalyzer()

    observed_max = (
        df[rating_col].max()
        if rating_col
        else None
    )

    text_scores = []
    emoji_scores = []
    rating_scores = []
    composites = []
    tiers = []
    emoji_lists = []

    for _, row in df.iterrows():

        text = row.get(
            text_col,
            ""
        )

        emojis_found = extract_emojis(text)

        t_score = text_sentiment_score(
            analyzer,
            text
        )

        e_score = emoji_sentiment_score(
            emojis_found
        )

        r_score = (
            normalize_rating(
                row.get(rating_col),
                observed_max
            )
            if rating_col
            else None
        )

        c_score = composite_score(
            r_score,
            t_score,
            e_score
        )

        text_scores.append(
            round(t_score, 4)
            if t_score is not None
            else None
        )

        emoji_scores.append(
            round(e_score, 4)
            if e_score is not None
            else None
        )

        rating_scores.append(
            round(r_score, 4)
            if r_score is not None
            else None
        )

        composites.append(c_score)

        tiers.append(
            tier_for_score(c_score)
        )

        emoji_lists.append(
            "".join(emojis_found)
            if emojis_found
            else ""
        )

    # Add analysis columns

    df["Emojis Found"] = emoji_lists

    df["Rating (normalized)"] = (
        rating_scores
    )

    df["Text Sentiment (0-1)"] = (
        text_scores
    )

    df["Emoji Sentiment (0-1)"] = (
        emoji_scores
    )

    df["Composite Score (0-1)"] = (
        composites
    )

    df["Feedback Tier"] = tiers

    # Sort by score

    df = df.sort_values(
        "Composite Score (0-1)",
        ascending=False
    ).reset_index(drop=True)

    # Add rank

    df.insert(
        0,
        "Rank",
        range(1, len(df) + 1)
    )

    return df, text_col, rating_col


# ============================================================
# EXCEL DOWNLOAD
# ============================================================

def create_excel_download(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Ranked Feedback"
        )

    output.seek(0)

    return output


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📊 Feedback Ranker")

st.write(
    "Upload your feedback Excel file and get "
    "automatically ranked feedback using text "
    "sentiment, emojis, and ratings."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload your Excel file",
    type=["xlsx", "xls"]
)


if uploaded_file is not None:

    try:

        df = pd.read_excel(
            uploaded_file
        )

        st.success(
            f"Successfully uploaded: {uploaded_file.name}"
        )

        st.write(
            f"📌 Total feedback entries: **{len(df)}**"
        )

        st.subheader(
            "📄 Uploaded Data Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.divider()

        # Analyze button

        if st.button(
            "🚀 Analyze & Rank Feedback",
            type="primary"
        ):

            with st.spinner(
                "Analyzing feedback..."
            ):

                try:

                    ranked_df, text_col, rating_col = (
                        analyze_feedback(df.copy())
                    )

                    st.success(
                        "✅ Feedback analysis completed!"
                    )

                    # --------------------------------
                    # Information
                    # --------------------------------

                    st.info(
                        f"📝 Feedback column detected: "
                        f"`{text_col}`"
                    )

                    if rating_col:

                        st.info(
                            f"⭐ Rating column detected: "
                            f"`{rating_col}`"
                        )

                    else:

                        st.info(
                            "⭐ No rating column detected. "
                            "Ranking was calculated using text "
                            "and emoji sentiment."
                        )

                    # --------------------------------
                    # Metrics
                    # --------------------------------

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Total Feedback",
                            len(ranked_df)
                        )

                    with col2:

                        best_count = (
                            ranked_df[
                                ranked_df[
                                    "Feedback Tier"
                                ] == "Best"
                            ].shape[0]
                        )

                        st.metric(
                            "🏆 Best",
                            best_count
                        )

                    with col3:

                        good_count = (
                            ranked_df[
                                ranked_df[
                                    "Feedback Tier"
                                ] == "Good"
                            ].shape[0]
                        )

                        st.metric(
                            "👍 Good",
                            good_count
                        )

                    with col4:

                        worst_count = (
                            ranked_df[
                                ranked_df[
                                    "Feedback Tier"
                                ] == "Worst"
                            ].shape[0]
                        )

                        st.metric(
                            "⚠️ Worst",
                            worst_count
                        )

                    st.divider()

                    # --------------------------------
                    # Results
                    # --------------------------------

                    st.subheader(
                        "🏆 Ranked Feedback"
                    )

                    st.dataframe(
                        ranked_df,
                        use_container_width=True,
                        height=500
                    )

                    # --------------------------------
                    # Tier breakdown
                    # --------------------------------

                    st.subheader(
                        "📊 Feedback Tier Breakdown"
                    )

                    tier_counts = (
                        ranked_df[
                            "Feedback Tier"
                        ]
                        .value_counts()
                        .reindex(
                            [
                                "Best",
                                "Good",
                                "Average",
                                "Poor",
                                "Worst"
                            ],
                            fill_value=0
                        )
                    )

                    st.bar_chart(
                        tier_counts
                    )

                    # --------------------------------
                    # Download
                    # --------------------------------

                    excel_file = (
                        create_excel_download(
                            ranked_df
                        )
                    )

                    st.download_button(
                        label="⬇️ Download Ranked Excel",
                        data=excel_file,
                        file_name=(
                            "feedback_ranked_output.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        )
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error while analyzing feedback: {e}"
                    )

    except Exception as e:

        st.error(
            f"❌ Could not read the uploaded Excel file: {e}"
        )


else:

    st.info(
        "👆 Upload an Excel file above to begin."
    )

    st.markdown(
        """
        ### How it works

        1. 📁 Upload your Excel feedback file
        2. 🔍 The system automatically detects the feedback column
        3. 😊 Text and emoji sentiment are analyzed
        4. ⭐ Ratings are incorporated when available
        5. 📊 A composite score is calculated
        6. 🏆 Feedback is ranked automatically
        7. ⬇️ Download the ranked Excel report
        """
    )