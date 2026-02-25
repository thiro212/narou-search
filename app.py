import streamlit as st
import requests
import pandas as pd
import gzip
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("なろう小説セマンティック検索")
st.write("読みたい内容を文章で入力すると、近い小説を探します！")

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def fetch_novels(genre, keyword, min_episodes):
    all_novels = []
    for page in range(1, 10):
        params = {
            "out": "json",
            "lim": 500,
            "st": (page - 1) * 500 + 1,
            "order": "hyoka",
            "of": "t-s-k-ga-n",
            "gzip": 5
        }
        if genre != 0:
            params["genre"] = genre
        if keyword:
            params["word"] = keyword

        try:
            response = requests.get("https://api.syosetu.com/novelapi/api/", params=params, timeout=30)
            data = json.loads(gzip.decompress(response.content))
            novels = data[1:]
            if not novels:
                break
            filtered = [n for n in novels if n.get("general_all_no", 0) >= min_episodes]
            all_novels.extend(filtered)
            if len(novels) < 500:
                break
        except:
            break
    if not all_novels:
        return pd.DataFrame()
    df = pd.DataFrame(all_novels)
    return df[["title", "story", "keyword", "general_all_no", "ncode"]].dropna()

# サイドバーに検索条件を配置
st.sidebar.title("検索条件")

genre_options = {
    "指定なし": 0,
    "異世界〔恋愛〕": 101,
    "現実世界〔恋愛〕": 102,
    "ハイファンタジー": 201,
    "ローファンタジー": 202,
    "純文学": 301,
    "ヒューマンドラマ": 302,
    "歴史": 303,
    "推理": 304,
    "ホラー": 305,
    "アクション": 306,
    "コメディー": 307,
    "VRゲーム": 401,
    "宇宙〔SF〕": 402,
    "空想科学": 403,
    "パニック": 404,
}

genre_label = st.sidebar.selectbox("ジャンル", list(genre_options.keys()), index=5)
genre = genre_options[genre_label]

keyword = st.sidebar.text_input("キーワード（空白でOK）", value="ダンジョン")
min_episodes = st.sidebar.slider("最低話数", 1, 500, 200)
top_k = st.sidebar.slider("表示件数", 5, 20, 10)

if st.sidebar.button("データを読み込む"):
    st.session_state["df"] = None
    st.session_state["embeddings"] = None

model = load_model()

if "df" not in st.session_state or st.session_state["df"] is None:
    with st.spinner("小説データを読み込み中..."):
        df = fetch_novels(genre, keyword, min_episodes)
        if df.empty:
            st.error("小説が見つかりませんでした。条件を変えてみてください。")
            st.stop()
        embeddings = model.encode(df["story"].tolist())
        st.session_state["df"] = df
        st.session_state["embeddings"] = embeddings

df = st.session_state["df"]
embeddings = st.session_state["embeddings"]

st.success(f"{len(df)}件の小説を読み込みました！")

query = st.text_input("読みたい内容を入力してください", placeholder="例：現実世界にダンジョンが現れて主人公が最強になっていく")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    st.write("### 検索結果")
    for i, idx in enumerate(top_indices):
        ncode = df['ncode'].iloc[idx]
        url = f"https://ncode.syosetu.com/{ncode.lower()}/"
        with st.expander(f"【{i+1}位】{df['title'].iloc[idx]}（{df['general_all_no'].iloc[idx]}話）"):
            st.write(f"類似度：{similarities[idx]:.3f}")
            st.write(f"あらすじ：{df['story'].iloc[idx]}")
            st.link_button("小説を読む", url)
