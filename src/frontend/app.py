import streamlit as st
import pandas as pd
import numpy as np
import os

# Configure the page
st.set_page_config(
    page_title="VibeMatch | Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Butterscotch / warm off-white background */
    .stApp {
        background-color: #fdf3e3;
        color: #2b2b2b;
    }

    /* Header */
    h1 {
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #845EC2, #D65DB1, #FF9671);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem !important;
        padding-bottom: 0rem !important;
    }

    /* Glassmorphic card */
    .glass-card {
        background: rgba(255, 248, 235, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 150, 80, 0.15);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(180, 120, 40, 0.06);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 252, 245, 0.92);
        box-shadow: 0 10px 20px rgba(180, 120, 40, 0.12);
        border: 1px solid rgba(200, 150, 80, 0.25);
    }

    /* Match score */
    .match-score {
        font-weight: 600;
        color: #845EC2;
        font-size: 1.1rem;
    }
    .live-badge {
        font-weight: 600;
        color: #c4621a;
        font-size: 1.1rem;
    }

    /* Track text */
    .track-name {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
        color: #2b2b2b;
    }
    .track-artist {
        font-size: 1rem;
        font-weight: 400;
        color: #7a5c3a;
        margin: 0;
    }

    /* Override Streamlit warning: dark text on amber */
    div[data-testid="stAlert"] {
        background-color: #fff3cd !important;
        border: 1px solid #e6a817 !important;
        border-radius: 10px !important;
        color: #5a3e00 !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: #5a3e00 !important;
    }

    /* Static music notes scattered across background */
    .music-notes-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .note {
        position: absolute;
        color: #c49a55;
        opacity: 0.18;
        user-select: none;
        font-style: normal;
    }
</style>
""", unsafe_allow_html=True)

# Static scattered music notes (fixed positions, no animation)
st.markdown("""
<div class="music-notes-bg">
  <span class="note" style="top:5%;  left:3%;  font-size:2.2rem;">♩</span>
  <span class="note" style="top:12%; left:91%; font-size:1.6rem;">♪</span>
  <span class="note" style="top:22%; left:8%;  font-size:1.3rem;">♫</span>
  <span class="note" style="top:18%; left:78%; font-size:2.4rem;">♬</span>
  <span class="note" style="top:35%; left:95%; font-size:1.8rem;">♩</span>
  <span class="note" style="top:40%; left:2%;  font-size:2rem;">♪</span>
  <span class="note" style="top:50%; left:88%; font-size:1.5rem;">♫</span>
  <span class="note" style="top:55%; left:14%; font-size:1.9rem;">♬</span>
  <span class="note" style="top:63%; left:72%; font-size:2.1rem;">♩</span>
  <span class="note" style="top:70%; left:6%;  font-size:1.4rem;">♪</span>
  <span class="note" style="top:75%; left:83%; font-size:1.7rem;">♫</span>
  <span class="note" style="top:82%; left:22%; font-size:2.3rem;">♬</span>
  <span class="note" style="top:88%; left:60%; font-size:1.6rem;">♩</span>
  <span class="note" style="top:93%; left:40%; font-size:1.8rem;">♪</span>
  <span class="note" style="top:8%;  left:50%; font-size:1.3rem;">♫</span>
  <span class="note" style="top:30%; left:35%; font-size:2rem;">♬</span>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sim_path     = os.path.join(base_dir, "models", "similarity_matrix.parquet")
    vectors_path = os.path.join(base_dir, "models", "song_vectors.parquet")

    if not os.path.exists(sim_path) or not os.path.exists(vectors_path):
        return None, None, None

    sim_df     = pd.read_parquet(sim_path)
    vectors_df = pd.read_parquet(vectors_path)

    # Build display name for dropdown — uses ALL songs from vectors file
    vectors_df["display_name"] = vectors_df["artist"] + " - " + vectors_df["song"]
    return sim_df, vectors_df


def live_recommend(selected_artist, selected_song, vectors_df, top_n=10):
    """Compute cosine similarity on-the-fly using pre-saved L2-normalised vectors."""
    feat_cols = [c for c in vectors_df.columns if c.startswith("f")]
    matrix = vectors_df[feat_cols].values.astype("float32")   # (N, 100)

    mask = (vectors_df["artist"] == selected_artist) & (vectors_df["song"] == selected_song)
    if not mask.any():
        return pd.DataFrame()

    query_vec = matrix[mask][0]                      # shape (100,)
    scores = matrix @ query_vec                      # cosine sim for all songs
    scores[mask] = -1                                # exclude self

    top_idx = np.argsort(scores)[::-1][:top_n]
    result = vectors_df.iloc[top_idx][["artist", "song"]].copy()
    result["distance"] = 1 - scores[top_idx]        # convert similarity → distance for display
    result["rank"] = range(1, top_n + 1)
    result = result.rename(columns={"song": "similar_song", "artist": "similar_artist"})
    return result.reset_index(drop=True)


def render_cards(recommendations, live=False):
    cols = st.columns(3)
    for i, (_, row) in enumerate(recommendations.iterrows()):
        col_idx = i % 3
        if live:
            # distance here = 1 - cosine_similarity
            score = max(0, min(100, int((1 - row["distance"]) * 100)))
            badge = f'<p class="live-badge">⚡ {score}% Live Match</p>'
        else:
            cos_sim = 1 - ((row["distance"] ** 2) / 2)
            score = max(0, min(100, int(cos_sim * 100)))
            badge = f'<p class="match-score">✨ {score}% Lyrical Match</p>'

        with cols[col_idx]:
            st.markdown(f"""
            <div class="glass-card">
                <p class="track-name">{row['similar_song']}</p>
                <p class="track-artist">👤 {row['similar_artist']}</p>
                <hr style="border-color: rgba(180,120,40,0.1); margin: 10px 0;">
                {badge}
            </div>
            """, unsafe_allow_html=True)


def main():
    st.markdown("<h1>VibeMatch 🎵</h1>", unsafe_allow_html=True)

    result = load_data()
    if result[0] is None:
        st.error("Data not found! Please run the PySpark backend training script first to generate the similarity matrix.")
        st.stop()

    sim_df, vectors_df = result

    cols = st.columns([1, 2, 1])
    with cols[1]:
        song_options = sorted(vectors_df["display_name"].unique())
        selected_display = st.selectbox("Search for a song you like:", ["Select a song..."] + song_options)

    if selected_display != "Select a song...":
        sel_row    = vectors_df[vectors_df["display_name"] == selected_display].iloc[0]
        sel_artist = sel_row["artist"]
        sel_song   = sel_row["song"]

        # Try pre-computed results first
        precomputed = sim_df[(sim_df["song"] == sel_song) & (sim_df["artist"] == sel_artist)]

        if not precomputed.empty:
            st.markdown("<br><h3 style='text-align:center;'>Top Recommendations based on Lyrical Vibe</h3><br>",
                        unsafe_allow_html=True)
            render_cards(precomputed.sort_values("rank"), live=False)
        else:
            # Fallback: compute live
            st.markdown("<br><h3 style='text-align:center;'>Top Recommendations ⚡ Computed Live</h3><br>",
                        unsafe_allow_html=True)
            with st.spinner("Computing similarity on-the-fly..."):
                live_recs = live_recommend(sel_artist, sel_song, vectors_df)

            if live_recs.empty:
                st.warning("Could not find this song in the feature database. Try another one.")
            else:
                render_cards(live_recs, live=True)


if __name__ == "__main__":
    main()
