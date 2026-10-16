"""
Smart Keyword Extractor
-----------------------
A beginner-friendly NLP project built with Streamlit and NLTK.
Keywords are found using simple word-frequency counting.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import html
from collections import Counter

import nltk
import streamlit as st
from nltk.tokenize import RegexpTokenizer

# ---------------------------------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Keyword Extractor",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 2. CUSTOM CSS (dark, clean theme injected with st.markdown)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Page background and centered container */
    .stApp {background-color: #0B0B0B; color: #FFFFFF;}
    .block-container {max-width: 820px; padding-top: 3rem; padding-bottom: 2rem;}

    /* Title area */
    .app-title {font-size: 2.4rem; font-weight: 700; color: #FFFFFF;
                margin-bottom: 0.2rem; letter-spacing: -0.5px;}
    .app-title span {color: #FF1E2D;}
    .app-subtitle {color: #A0A0A0; font-size: 1.05rem; margin-bottom: 2rem;}

    /* Text area */
    .stTextArea textarea {background-color: #151515; color: #FFFFFF;
        border: 1px solid #2A2A2A; border-radius: 10px; font-size: 1rem;}
    .stTextArea textarea:focus {border-color: #FF1E2D; box-shadow: none;}
    .stTextArea label, .stSelectbox label {color: #A0A0A0 !important;}

    /* Select box */
    div[data-baseweb="select"] > div {background-color: #151515;
        border: 1px solid #2A2A2A; color: #FFFFFF;}

    /* Primary button */
    .stButton > button {background-color: #FF1E2D; color: #FFFFFF; border: none;
        border-radius: 8px; padding: 0.65rem 1.5rem; font-weight: 600; width: 100%;}
    .stButton > button:hover {background-color: #D91421; color: #FFFFFF; border: none;}
    .stButton > button:focus {box-shadow: none; color: #FFFFFF;}

    /* Small text under the text area */
    .counter {color: #A0A0A0; font-size: 0.85rem; margin-top: -0.5rem;
              margin-bottom: 1rem;}

    /* Statistic cards */
    .stat-row {display: flex; gap: 12px; margin: 1.5rem 0;}
    .stat-card {flex: 1; background: #151515; border: 1px solid #2A2A2A;
                border-radius: 10px; padding: 1rem; text-align: center;}
    .stat-value {font-size: 1.7rem; font-weight: 700; color: #FF1E2D;}
    .stat-label {font-size: 0.75rem; color: #A0A0A0; text-transform: uppercase;
                 letter-spacing: 1px;}

    /* Keyword result cards */
    .section-label {color: #A0A0A0; font-size: 0.8rem; letter-spacing: 2px;
                    text-transform: uppercase; margin: 1.5rem 0 0.8rem 0;}
    .kw-card {background: #151515; border: 1px solid #2A2A2A; border-radius: 10px;
              padding: 0.8rem 1rem; margin-bottom: 8px;}
    .kw-top {display: flex; align-items: center; justify-content: space-between;}
    .kw-rank {color: #FF1E2D; font-weight: 700; width: 3rem;}
    .kw-word {flex: 1; font-size: 1.1rem; font-weight: 600; color: #FFFFFF;}
    .kw-count {color: #A0A0A0; font-size: 0.9rem;}
    .kw-bar-bg {background: #2A2A2A; height: 4px; border-radius: 2px; margin-top: 8px;}
    .kw-bar {background: #FF1E2D; height: 4px; border-radius: 2px;}

    /* Expander */
    div[data-testid="stExpander"] {background: #151515; border: 1px solid #2A2A2A;
                                   border-radius: 10px;}
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] li {color: #D0D0D0;}

    /* Footer */
    .footer {text-align: center; color: #A0A0A0; font-size: 0.85rem;
             border-top: 1px solid #2A2A2A; margin-top: 3rem; padding-top: 1.2rem;
             line-height: 1.6;}
    .footer b {color: #FFFFFF;}

    /* Mobile layout */
    @media (max-width: 600px) {
        .app-title {font-size: 1.8rem;}
        .stat-row {flex-direction: column;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3. NLTK STOPWORDS (downloaded automatically, with a safe fallback)
# ---------------------------------------------------------------------------
# Small built-in list used only if the NLTK download fails (e.g. no internet).
FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "she",
    "so", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "was", "we", "were", "what", "when", "which", "who",
    "will", "with", "you", "your", "can", "do", "does", "did", "also", "about",
    "just", "very", "more", "most", "some", "such", "only", "other", "any",
}


@st.cache_resource
def load_stopwords():
    """Return (stopword_set, using_nltk). Never crashes if NLTK data is missing."""
    try:
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english")), True
    except Exception:
        return FALLBACK_STOPWORDS, False


STOP_WORDS, USING_NLTK = load_stopwords()

# This tokenizer keeps only letters, so punctuation and numbers are
# removed automatically. It needs no downloaded NLTK data.
tokenizer = RegexpTokenizer(r"[a-z]+")


# ---------------------------------------------------------------------------
# 4. KEYWORD EXTRACTION LOGIC
# ---------------------------------------------------------------------------
def extract_keywords(text, top_n):
    """Return (keywords, total_words, unique_words).

    keywords is a list of (word, count) sorted by count (highest first).
    """
    # Step 1: lowercase
    text = text.lower()

    # Step 2 + 3: remove punctuation and tokenize into words
    all_words = tokenizer.tokenize(text)

    # Step 4 + 5: remove stopwords and very short words (< 3 letters)
    filtered = [w for w in all_words if w not in STOP_WORDS and len(w) >= 3]

    # Step 6: count how often each remaining word appears
    counts = Counter(filtered)

    # Step 7: sort by frequency and keep the top N
    keywords = counts.most_common(top_n)

    return keywords, len(all_words), len(set(all_words))


# ---------------------------------------------------------------------------
# 5. USER INTERFACE
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-title">Smart <span>Keyword</span> Extractor</div>'
    '<div class="app-subtitle">Extract meaningful keywords from any text using NLP</div>',
    unsafe_allow_html=True,
)

# Remember whether the button was clicked, so results stay when
# the user changes the keyword count.
if "extracted" not in st.session_state:
    st.session_state.extracted = False

text_input = st.text_area(
    "Paste your text here",
    height=240,
    placeholder="Paste an article, essay, or paragraph here...",
)

# Live character / word count
char_count = len(text_input)
word_count = len(text_input.split())
st.markdown(
    f'<div class="counter">{char_count} characters &nbsp;|&nbsp; {word_count} words</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 2])
with col1:
    top_n = st.selectbox("Number of keywords", [5, 10, 15, 20], index=1)
with col2:
    st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
    if st.button("Extract Keywords"):
        st.session_state.extracted = True

# ---------------------------------------------------------------------------
# 6. SHOW RESULTS (with error handling)
# ---------------------------------------------------------------------------
if st.session_state.extracted:
    if not text_input.strip():
        # Empty text box
        st.warning("Please paste some text first, then click Extract Keywords.")
    elif word_count < 5:
        # Very short text
        st.info("Your text is very short. Please enter at least a few sentences "
                "to get meaningful keywords.")
    else:
        keywords, total_words, unique_words = extract_keywords(text_input, top_n)

        if not keywords:
            st.info("No keywords found. The text may contain only common words "
                    "(like 'the' or 'and') or very short words. Try a longer text.")
        else:
            # Summary statistics
            st.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-card"><div class="stat-value">{total_words}</div>
                        <div class="stat-label">Total Words</div></div>
                    <div class="stat-card"><div class="stat-value">{unique_words}</div>
                        <div class="stat-label">Unique Words</div></div>
                    <div class="stat-card"><div class="stat-value">{len(keywords)}</div>
                        <div class="stat-label">Keywords Extracted</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Keyword cards
            st.markdown('<div class="section-label">Keywords Found</div>',
                        unsafe_allow_html=True)

            highest = keywords[0][1]  # used to size the bars
            cards = ""
            for rank, (word, count) in enumerate(keywords, start=1):
                width = int(count / highest * 100)
                label = "occurrence" if count == 1 else "occurrences"
                cards += f"""
                <div class="kw-card">
                    <div class="kw-top">
                        <span class="kw-rank">#{rank}</span>
                        <span class="kw-word">{html.escape(word.capitalize())}</span>
                        <span class="kw-count">{count} {label}</span>
                    </div>
                    <div class="kw-bar-bg"><div class="kw-bar" style="width:{width}%"></div></div>
                </div>
                """
            st.markdown(cards, unsafe_allow_html=True)

            if not USING_NLTK:
                st.caption("Note: NLTK stopwords could not be downloaded, "
                           "so a built-in stopword list was used instead.")

# ---------------------------------------------------------------------------
# 7. EDUCATIONAL SECTION
# ---------------------------------------------------------------------------
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
with st.expander("How does it work?"):
    st.markdown(
        """
1. **Text preprocessing** - The text is converted to lowercase so that
   "Python" and "python" are treated as the same word. Punctuation and numbers are removed.
2. **Tokenization** - The cleaned text is split into individual words (tokens)
   using NLTK's tokenizer.
3. **Stop-word removal** - Very common words such as "the", "is" and "and" carry
   little meaning, so they are removed. Words shorter than 3 letters are removed too.
4. **Frequency calculation** - The remaining words are counted to see how many
   times each one appears in the text.
5. **Keyword ranking** - Words are sorted from most to least frequent. The words at the
   top of the list are the keywords, because words repeated often are usually the main topic.
        """
    )

# ---------------------------------------------------------------------------
# 8. FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Developed by <b>Makwana Dixit </b><br>
        Information &amp; Communication Technology (ICT)<br>
        Dr. Jivraj N. Mehta Government Polytechnic, Amreli
    </div>
    """,
    unsafe_allow_html=True,
)
