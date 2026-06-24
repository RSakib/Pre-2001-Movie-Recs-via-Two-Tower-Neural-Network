import os
import time
import pandas as pd
import numpy as np
import gradio as gr
import faiss
from sentence_transformers import SentenceTransformer

# Clean, production-safe single-line CSS properties
netflix_css = (
    "body, .gradio-container, html { background-color: #000000 !important; color: #FFFFFF !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important; }\n"
    ".gradio-container .block, .gradio-container .form, .gradio-container .fieldset, .gradio-container .gr-box, div[class*='form'], div[class*='fieldset'], div[class*='block'] { background: #000000 !important; background-color: #000000 !important; border: none !important; box-shadow: none !important; padding: 0 !important; margin-bottom: 12px !important; }\n"
    ".gradio-container select, .gradio-container .dropdown-container, .gradio-container div[role='listbox'] { background-color: #000000 !important; background: #000000 !important; color: #FFFFFF !important; border: 1px solid #333333 !important; border-radius: 4px !important; padding: 6px 10px !important; font-size: 9pt !important; }\n"
    ".gradio-container select:focus { border-color: #E50914 !important; outline: none !important; }\n"
    "button.primary, .primary-btn { background-color: #E50914 !important; color: #FFFFFF !important; border: none !important; font-weight: bold !important; text-transform: uppercase !important; letter-spacing: 1px !important; transition: background-color 0.2s ease, transform 0.1s ease !important; border-radius: 4px !important; padding: 10px 16px !important; width: 100% !important; cursor: pointer !important; margin-top: 5px !important; font-size: 9.5pt !important; }\n"
    "button.primary:hover, .primary-btn:hover { background-color: #B80710 !important; transform: scale(1.01) !important; }\n"
    ".netflix-vertical-scroll { max-height: 520px !important; overflow-y: auto !important; overflow-x: visible !important; padding: 12px !important; }\n"
    ".netflix-vertical-scroll::-webkit-scrollbar { width: 6px !important; }\n"
    ".netflix-vertical-scroll::-webkit-scrollbar-track { background: #000000 !important; }\n"
    ".netflix-vertical-scroll::-webkit-scrollbar-thumb { background: #222222 !important; border-radius: 3px !important; }\n"
    ".netflix-vertical-scroll::-webkit-scrollbar-thumb:hover { background: #E50914 !important; }\n"
    ".netflix-grid-3 { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 14px !important; width: 100% !important; }\n"
    ".netflix-card { background-color: #141414 !important; border-radius: 4px !important; overflow: hidden !important; transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.25s ease, border-color 0.25s ease !important; border: 1px solid #222222 !important; position: relative !important; aspect-ratio: 2 / 3 !important; width: 100% !important; }\n"
    ".netflix-card:hover { transform: scale(1.08) !important; box-shadow: 0 10px 20px rgba(229, 9, 20, 0.5) !important; border-color: #E50914 !important; z-index: 9999 !important; }\n"
    ".card-poster { width: 100% !important; height: 100% !important; object-fit: cover !important; display: block !important; }\n"
    ".card-overlay { position: absolute !important; bottom: 0 !important; left: 0 !important; right: 0 !important; background: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0.85) 65%, rgba(0,0,0,0) 100%) !important; padding: 8px 6px !important; z-index: 2 !important; opacity: 0 !important; transition: opacity 0.25s ease !important; }\n"
    ".netflix-card:hover .card-overlay { opacity: 1 !important; }\n"
    ".card-title { font-size: 9pt !important; font-weight: 700 !important; line-height: 1.15 !important; margin-bottom: 2px !important; color: #FFFFFF !important; display: -webkit-box !important; -webkit-line-clamp: 2 !important; -webkit-box-orient: vertical !important; overflow: hidden !important; }\n"
    ".card-genres { font-size: 6.5pt !important; color: #E50914 !important; text-transform: uppercase !important; letter-spacing: 0.3px !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }\n"
    ".gradio-container label span { color: #999999 !important; font-size: 8.5pt !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; margin-bottom: 4px !important; background-color: transparent !important; }\n"
    "/* Optimized 3x3 Grid Layout Configuration for 9 Selection Items */\n"
    ".taste-container-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 6px !important; max-width: 270px !important; margin-bottom: 12px !important; }\n"
    ".taste-mini-card { position: relative !important; aspect-ratio: 2 / 3 !important; border: 1.5px solid #262626 !important; border-radius: 3px !important; overflow: hidden !important; cursor: pointer !important; transition: all 0.2s ease !important; }\n"
    ".taste-mini-card:hover { border-color: #555555 !important; transform: scale(1.03) !important; }\n"
    ".taste-mini-card.active { border-color: #E50914 !important; box-shadow: 0 0 8px rgba(229, 9, 20, 0.7) !important; }\n"
    ".taste-mini-img { width: 100% !important; height: 100% !important; object-fit: cover !important; pointer-events: none !important; }\n"
    ".taste-overlay { position: absolute !important; bottom: 0 !important; left: 0 !important; right: 0 !important; background: linear-gradient(to top, rgba(0,0,0,1) 0%, rgba(0,0,0,0.9) 85%, rgba(0,0,0,0) 100%) !important; padding: 4px 3px !important; z-index: 2 !important; opacity: 0 !important; transition: opacity 0.2s ease !important; pointer-events: none !important; }\n"
    ".taste-mini-card:hover .taste-overlay { opacity: 1 !important; }\n"
    ".taste-title { font-size: 5.5pt !important; font-weight: 700 !important; line-height: 1.1 !important; color: #FFFFFF !important; display: -webkit-box !important; -webkit-line-clamp: 3 !important; -webkit-box-orient: vertical !important; overflow: hidden !important; text-align: center !important; }\n"
    "/* Mask the textbox component safely while keeping it fully initialized in the browser DOM */\n"
    "#hidden_taste_input, #hidden_eval_trigger { display: none !important; }\n"
    "/* Custom style wrapper for overlay evaluation box layout - stark black background */\n"
    ".eval-modal-container { background-color: #000000 !important; border: 1px solid #E50914 !important; border-radius: 6px !important; padding: 16px !important; margin-bottom: 20px !important; }\n"
    "/* Style target to cleanly size down and center the embedded t-SNE component frame */\n"
    ".scaled-plot-box { max-width: 75% !important; margin: 0 auto 12px auto !important; }\n"
)

try:
    model = SentenceTransformer("data/two_tower_model")
    index = faiss.read_index("data/movies.index")
    movies_df = pd.read_csv("data/processed_movies.csv")
    load_error_msg = None
except Exception as e:
    model = None
    index = None
    movies_df = None
    load_error_msg = f"CRITICAL SYSTEM ERROR: Failed to load pipeline artifacts. Details: {str(e)}"

fallback_img = "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=120&auto=format&fit=crop&q=60"

# --- MATHEMATICAL DATA POPULARITY ENGINE ---
TARGET_TITLES = []
if movies_df is not None:
    ratings_path = os.path.join("data", "ratings.csv")
    movies_raw_path = os.path.join("data", "movies.csv")
    
    if os.path.exists(ratings_path) and os.path.exists(movies_raw_path):
        try:
            ratings_df = pd.read_csv(ratings_path)
            raw_movies_df = pd.read_csv(movies_raw_path)
            
            counts = ratings_df['movie_id'].value_counts().reset_index()
            counts.columns = ['movie_id', 'interaction_count']
            
            merged_influence = raw_movies_df.merge(counts, on='movie_id')
            top_9_movies = merged_influence.sort_values(by='interaction_count', ascending=False).head(9)
            TARGET_TITLES = top_9_movies['movie_title'].str.strip().tolist()
        except Exception:
            TARGET_TITLES = []

if not TARGET_TITLES or len(TARGET_TITLES) < 9:
    TARGET_TITLES = [
        "Toy Story (1995)", "Matrix, The (1999)", "Godfather, The (1972)",
        "Pulp Fiction (1994)", "Star Wars: Episode IV - A New Hope (1977)", "Lion King, The (1994)",
        "Jurassic Park (1993)", "Forrest Gump (1994)", "Shining, The (1980)"
    ]
    TARGET_TITLES = TARGET_TITLES[:9]


def build_taste_grid_html():
    """Generates a structured 3x3 layout configuration mapping out the anchor movies."""
    html = "<div class='taste-container-grid'>"
    for target in TARGET_TITLES:
        poster_url = fallback_img
        if movies_df is not None:
            matched_rows = movies_df[movies_df['movie_title'].str.strip() == target]
            if not matched_rows.empty:
                found_url = str(matched_rows.iloc[0].get('poster_url', ''))
                if found_url and found_url != 'nan' and found_url.strip() != '':
                    poster_url = found_url
                    
        safe_title = target.replace("'", "\\'")
        
        html += (
            f"<div class='taste-mini-card' data-title='{safe_title}' "
            f"onclick=\"this.classList.toggle('active'); "
            f"let selected = Array.from(document.querySelectorAll('.taste-mini-card.active')).map(c => c.getAttribute('data-title')); "
            f"let el = document.querySelector('#hidden_taste_input textarea'); "
            f"if(el) {{ el.value = selected.join('|||'); el.dispatchEvent(new Event('input', {{ bubbles: true }})); }}\">"
            f"<img class='taste-mini-img' src='{poster_url}' />"
            f"<div class='taste-overlay'><div class='taste-title'>{target}</div></div>"
            f"</div>"
        )
    html += "</div>"
    return html

def recommend_movies(gender, selected_age_range, raw_favorites_str):
    if load_error_msg is not None or model is None or index is None or movies_df is None:
        raise gr.Error("Two-Tower models or FAISS indexes failed to load. Check server terminal.")

    # --- MATRICULATION DICTIONARY ---
    age_mapping = {
        "1-17": "1",
        "18-24": "18",
        "25-34": "25",
        "35-44": "35",
        "45-49": "45",
        "50-55": "50",
        "56+": "56"
    }
    
    age_str = age_mapping.get(str(selected_age_range).strip(), "25")
    gender_str = "Male" if str(gender).strip().upper() == "M" else "Female"

    user_profile_text = f"User Profile: {gender_str}, Age Group: {age_str}"
    
    raw_favorites = []
    if raw_favorites_str and raw_favorites_str.strip() != "":
        raw_favorites = [t.strip() for t in raw_favorites_str.split("|||") if t.strip() != ""]
        clean_favorites = []
        for t in raw_favorites:
            if " (" in t:
                clean_favorites.append(t.split(" (")[0])
            else:
                clean_favorites.append(t)
                
        if clean_favorites:
            favorites_str = ", ".join(clean_favorites)
            user_profile_text += f", Favorite Anchor References: {favorites_str}"
    
    user_vector = model.encode([user_profile_text]).astype('float32')
    faiss.normalize_L2(user_vector)
    distances, indices = index.search(user_vector, 15)
    candidates = movies_df.iloc[indices[0]].copy()

    if raw_favorites:
        candidates = candidates[~candidates['movie_title'].str.strip().isin(raw_favorites)]
        
    recommendations = candidates.head(12)

    unique_id = f"feed-{int(time.time() * 1000)}"
    html_out = f"<div id='{unique_id}' class='netflix-vertical-scroll'><div class='netflix-grid-3'>"

    for _, row in recommendations.iterrows():
        title = str(row['movie_title'])
        genres = str(row.get('movie_genres', row.get('genres', 'General')))
        poster_url = str(row.get('poster_url', ''))
        
        if not poster_url or poster_url == 'nan' or poster_url.strip() == '':
            poster_url = fallback_img

        html_out += f"<div class='netflix-card'><img class='card-poster' src='{poster_url}' alt='{title}' /><div class='card-overlay'><div class='card-title'>{title}</div><div class='card-genres'>{genres}</div></div></div>"
        
    html_out += "</div></div>"
    return html_out

def clear_feed():
    return ""

def open_modal():
    return gr.update(visible=True)

def close_modal():
    return gr.update(visible=False)

# Top header block featuring absolute-positioned evaluation trigger link on the top right
header_html = (
    "<div style='position: relative; text-align: center; padding: 10px 0 15px 0;'>\n"
    "    <a href='javascript:void(0)' onclick=\"document.getElementById('hidden_eval_trigger').click()\" style='position: absolute; top: 10px; right: 10px; color: #E50914; font-size: 9.5pt; text-decoration: underline; font-weight: bold;'>Model Evaluation ?</a>\n"
    "    <h1 style='color: #E50914; font-size: 14pt; font-weight: 800; letter-spacing: -0.2px; margin-bottom: 2px; text-transform: uppercase;'>Movie Recs from Pre-2001</h1>\n"
    "    <p style='color: #999999; font-size: 8.5pt; margin-top: 0; margin-bottom: 0;'>Using a Two-Tower Neural Network</p>\n"
    "</div>"
)

with gr.Blocks(css=netflix_css, title="Movie Recs from Pre-2001") as demo:
    gr.HTML(header_html)
    
    # Kept visible=True for DOM discovery, but cleanly masked via netflix_css (#hidden_eval_trigger)
    eval_trigger_btn = gr.Button(visible=True, elem_id="hidden_eval_trigger")
    
    # Custom Layout Column container - stark black background wrapper configuration
    with gr.Column(visible=False, elem_classes="eval-modal-container") as eval_modal:
        gr.HTML(
            "<h2 style='font-size: 18pt; font-weight: bold; color: #FFFFFF; margin: 0 0 4px 0; line-height: 1.1;'>Model Evaluation Summary</h2>"
            "<h3 style='font-size: 13pt; font-weight: bold; color: #FFFFFF; margin: 0 0 4px 0; line-height: 1.1;'>"
            "This model achieved an NDCG@10 relevance score of 0.032."
            "</h3>"
            "<p style='font-size: 9.5pt; color: #999999; margin: 0 0 16px 0; line-height: 1.4; text-align: justify; max-width: 95%;'>"
            "This evaluation was done on an 80/20 time-based split across all movies that scored above a 4.0 with somebody. "
            "It was trained my single machine with 75000 samples and 5 epochs with a batch size of 64. "
            "The score can go higher with more samples and a bigger batch size, but it would take very long on my local machine. "
            "If I wanted to increase the score higher with more resources, I would train with higher numbers across the board to "
            "get a better relevance score on this neural network"
            "</p>"
            "<h3 style='font-size: 14pt; font-weight: bold; color: #FFFFFF; margin: 20px 0 12px 0;'>"
            "Embedding Space Visualization"
            "</h3>"
        )
        
        # Displaying the local embedding visualization plot styled down smaller
        with gr.Column(elem_classes="scaled-plot-box"):
            gr.Image(value="movie_embeddings_tsne.png", show_label=False, container=False)
            gr.HTML(
                "<p style='font-size: 9.5pt; color: #999999; margin: 8px 0 0 0; text-align: justify; line-height: 1.3;'>"
                "This is a visualization of how my neural network clusters movies, which are color coded by their genre. "
                "My network has the well-aligned embeddings because this visualization shows that movies of certain genres "
                "are being grouped with similar genres."
                "</p>"
            )
            
        close_btn = gr.Button("Dismiss Window", size="sm")

    # Hooking up modal interaction state pipelines
    eval_trigger_btn.click(fn=open_modal, inputs=None, outputs=eval_modal)
    close_btn.click(fn=close_modal, inputs=None, outputs=eval_modal)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h4 style='color: #FFFFFF; border-left: 3px solid #E50914; padding-left: 6px; margin-bottom: 12px; font-size: 10pt; text-transform: uppercase; letter-spacing: 0.5px;'>Create your User Profile</h4>")
            gender = gr.Dropdown(choices=["M", "F"], value="M", label="Gender", type="value")
            
            age = gr.Dropdown(
                choices=["1-17", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"], 
                value="25-34", 
                label="Age Group", 
                type="value"
            )
            
            gr.Markdown("<h4 style='color: #FFFFFF; border-left: 3px solid #E50914; padding-left: 6px; margin-top: 12px; margin-bottom: 8px; font-size: 8.5pt; text-transform: uppercase; letter-spacing: 0.5px;'>Select Movies that you Like</h4>")
            
            gr.HTML(value=build_taste_grid_html())
            hidden_favorites = gr.Textbox(visible=True, elem_id="hidden_taste_input", value="")
            
            submit_btn = gr.Button("Generate Feed", elem_classes="primary-btn", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("<h4 style='color: #FFFFFF; border-left: 3px solid #E50914; padding-left: 6px; margin-bottom: 12px; font-size: 10pt; text-transform: uppercase; letter-spacing: 0.5px;'>Personalized Movie Recommendations</h4>")
            output_html = gr.HTML(value="", label="Output Viewport")
            
    submit_btn.click(fn=clear_feed, inputs=None, outputs=output_html).then(
        fn=recommend_movies, inputs=[gender, age, hidden_favorites], outputs=output_html
    )

if __name__ == "__main__":
    demo.launch()