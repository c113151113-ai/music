"""
Chapter 6: 以 Streamlit 建立使用者介面

直接重用 Chapter 4 建好的 msd.db 與 Chapter 5 的搜尋 / 推薦演算法
(chapter5_music_api.py)，提供網頁操作介面：
  🔍 搜尋歌曲      (FTS5 全文檢索)
  🔥 熱門歌曲      (依聽眾數 / 播放數)
  🎧 歌曲推薦      (物品協同過濾 + 內容導向)
  👤 使用者推薦    (使用者協同過濾 vs 熱門基準線)
  📊 資料概覽

安裝與執行 (本程式需與 msd.db、chapter5_music_api.py 放在同一資料夾):
  pip install streamlit pandas numpy fastapi uvicorn
  python chapter5_music_api.py --prepare      # 只需做一次 (若已做過可略過)
  streamlit run chapter6_streamlit_ui.py
"""

import os
import random
import sqlite3
import sys
from contextlib import contextmanager

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

st.set_page_config(page_title="音樂搜尋與推薦", page_icon="🎵", layout="wide")

# ------------------------------------------------------------------
# 載入 Chapter 5 的演算法
# ------------------------------------------------------------------
try:
    import chapter5_music_api as c5
    from fastapi import HTTPException
except ImportError as e:
    st.error(f"無法載入 chapter5_music_api.py 或其套件：{e}\n\n"
             "請確認檔案在同一資料夾，並執行 `pip install fastapi uvicorn numpy pandas`。")
    st.stop()

if not os.path.isfile(c5.DB_PATH):
    st.error(f"找不到資料庫：{c5.DB_PATH}\n請先完成 Chapter 4 產生 msd.db。")
    st.stop()


@contextmanager
def db():
    """每次查詢開一條唯讀連線，避免 Streamlit 多執行緒共用連線的問題。"""
    conn = sqlite3.connect(c5.DB_PATH)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA cache_size = -100000")
    try:
        yield conn
    finally:
        conn.close()


# 檢查 --prepare 是否已執行
with db() as _conn:
    _missing = [t for t in ("song_stats", "listening_filtered", "listening_rated")
                if not c5.has_table(_conn, t)]
    _has_fts = c5.has_table(_conn, "songs_fts")
if _missing:
    st.error(f"資料庫缺少資料表 {_missing}。\n\n"
             "請先在終端機執行：`python chapter5_music_api.py --prepare`")
    st.stop()


@st.cache_resource(show_spinner="載入音訊特徵矩陣…")
def init_content() -> int:
    try:
        c5.load_content_matrix()
    except Exception:  # spotify_tracks 不存在或沒資料時，內容導向功能停用
        c5.STATE.update(ids=[], pos={}, matrix=None)
    return len(c5.STATE.get("ids", []))


n_audio = init_content()

# ------------------------------------------------------------------
# 資料存取（結果快取）。回傳一律是 list[dict]，方便快取與顯示
# ------------------------------------------------------------------
COLS = {
    "song_id": "歌曲ID", "title": "歌名", "artist_name": "歌手", "year": "年份",
    "total_plays": "總播放", "listeners": "聽眾數",
    "has_audio_features": "有音訊特徵", "score": "分數",
}


def dump(s) -> dict:
    return s.model_dump() if hasattr(s, "model_dump") else s.dict()


def safe(fn):
    """執行推薦函式，把 HTTPException 轉成錯誤訊息。回傳 (rows, error)。"""
    try:
        return [dump(s) for s in fn()], None
    except HTTPException as e:
        return [], e.detail


@st.cache_data(show_spinner="搜尋中…")
def search_songs(q: str, limit: int):
    match = c5.fts_query(q)
    if not match:
        return [], None
    with db() as conn:
        if _has_fts:
            rows = conn.execute(
                f"SELECT {c5.SONG_COLS} FROM songs_fts JOIN songs s ON s.rowid = songs_fts.rowid "
                f"LEFT JOIN song_stats st ON st.song_id = s.song_id "
                f"WHERE songs_fts MATCH ? ORDER BY COALESCE(st.listeners, 0) DESC LIMIT ?",
                (match, limit)).fetchall()
        else:
            like = f"%{q}%"
            rows = conn.execute(
                f"SELECT {c5.SONG_COLS} FROM {c5.SONG_FROM} "
                f"WHERE s.title LIKE ? OR s.artist_name LIKE ? "
                f"ORDER BY COALESCE(st.listeners, 0) DESC LIMIT ?", (like, like, limit)).fetchall()
    return [dump(c5.to_song(r)) for r in rows], None


@st.cache_data(show_spinner="查詢中…")
def get_popular(limit: int, by: str):
    with db() as conn:
        return safe(lambda: c5.popular_songs(conn, limit, by))


@st.cache_data(show_spinner="計算物品協同過濾…")
def get_item_cf(song_id: str, limit: int):
    with db() as conn:
        return safe(lambda: c5.ranked_songs(conn, c5.item_cf(conn, song_id, limit)))


@st.cache_data(show_spinner="計算內容導向推薦…")
def get_content(song_id: str, limit: int):
    with db() as conn:
        return safe(lambda: c5.ranked_songs(conn, c5.content_similar(song_id, limit)))


@st.cache_data(show_spinner="計算使用者協同過濾…")
def get_user_cf(user_idx: int, limit: int):
    with db() as conn:
        return safe(lambda: c5.ranked_songs(conn, c5.user_cf(conn, user_idx, limit)))


@st.cache_data(show_spinner=False)
def get_history(user_idx: int, limit: int):
    with db() as conn:
        def run():
            rows = conn.execute(
                "SELECT song_id, rating FROM listening_rated WHERE user_idx = ? "
                "ORDER BY rating DESC LIMIT ?", (user_idx, limit)).fetchall()
            if not rows:
                raise HTTPException(404, "找不到此使用者（可能已被過濾條件排除）")
            return c5.ranked_songs(conn, rows)
        return safe(run)


@st.cache_data(show_spinner="統計中…")
def get_overview():
    with db() as conn:
        q = conn.execute
        info = {
            "歌曲數": q("SELECT COUNT(*) FROM songs").fetchone()[0],
            "有聆聽紀錄的歌曲": q("SELECT COUNT(*) FROM song_stats").fetchone()[0],
            "過濾後使用者數": q("SELECT COUNT(DISTINCT user_idx) FROM listening_rated").fetchone()[0],
            "過濾後聆聽紀錄": q("SELECT COUNT(*) FROM listening_filtered").fetchone()[0],
        }
        sp = q("SELECT COUNT(*), SUM(song_id IS NOT NULL) FROM spotify_tracks").fetchone()
        info["Spotify 曲目 / 已對應 MSD"] = f"{sp[0]:,} / {int(sp[1] or 0):,}"
        years = pd.read_sql_query(
            "SELECT year, COUNT(*) AS n FROM songs WHERE year IS NOT NULL AND year > 0 "
            "GROUP BY year ORDER BY year", conn)
    return info, years


def random_user() -> int:
    with db() as conn:
        mx = conn.execute("SELECT MAX(rowid) FROM listening_rated").fetchone()[0] or 1
        row = conn.execute("SELECT user_idx FROM listening_rated WHERE rowid >= ? LIMIT 1",
                           (random.randint(1, mx),)).fetchone()
    return int(row[0]) if row else 0


def to_df(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows).rename(columns=COLS)


def label(s: dict) -> str:
    yr = f" ({s['year']})" if s.get("year") else ""
    return f"{s['artist_name']} — {s['title']}{yr}"


def show_songs(rows: list, empty_msg: str = "沒有結果") -> None:
    if not rows:
        st.info(empty_msg)
        return
    df = to_df(rows)
    show = [c for c in ["歌手", "歌名", "年份", "聽眾數", "總播放", "分數", "有音訊特徵"] if c in df.columns]
    st.dataframe(df[show], hide_index=True)


# ------------------------------------------------------------------
# 側邊欄
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 系統狀態")
st.sidebar.write(f"資料庫：`{os.path.basename(c5.DB_PATH)}`")
st.sidebar.write(f"全文檢索 (FTS5)：{'✅ 啟用' if _has_fts else '⚠️ 未建立，改用 LIKE (較慢)'}")
st.sidebar.write(f"音訊特徵歌曲：{n_audio:,} 首")
result_limit = st.sidebar.slider("推薦 / 結果筆數", 5, 50, 10)
if st.sidebar.button("清除快取"):
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------------------------------
# 主畫面
# ------------------------------------------------------------------
st.title("🎵 音樂搜尋與推薦系統")
st.caption("Million Song Dataset + Taste Profile + Spotify 音訊特徵｜以 Streamlit 建立的使用者介面")

tab_search, tab_pop, tab_song, tab_user, tab_info = st.tabs(
    ["🔍 搜尋歌曲", "🔥 熱門歌曲", "🎧 歌曲推薦", "👤 使用者推薦", "📊 資料概覽"])

# ----------------------------- 搜尋 -----------------------------
with tab_search:
    with st.form("search_form"):
        c1, c2 = st.columns([4, 1])
        q = c1.text_input("歌名或歌手", placeholder="例如：yellow、coldplay")
        lim = c2.number_input("筆數", 5, 100, 20, step=5)
        submitted = st.form_submit_button("搜尋", type="primary")
    if submitted:
        if not q.strip():
            st.warning("請輸入關鍵字")
        else:
            rows, _ = search_songs(q.strip(), int(lim))
            st.success(f"找到 {len(rows)} 首（依聽眾數排序）")
            show_songs(rows, "找不到符合的歌曲")

# ----------------------------- 熱門 -----------------------------
with tab_pop:
    by_label = st.radio("排序依據", ["聽眾數", "總播放次數"], horizontal=True)
    by = "listeners" if by_label == "聽眾數" else "plays"
    rows, err = get_popular(result_limit, by)
    if err:
        st.error(err)
    else:
        show_songs(rows)
        df = to_df(rows)
        df["顯示名稱"] = df["歌手"] + " — " + df["歌名"]
        st.bar_chart(df.set_index("顯示名稱")["聽眾數" if by == "listeners" else "總播放"])

# --------------------------- 歌曲推薦 ---------------------------
with tab_song:
    st.write("先搜尋一首歌，再看「聽過這首歌的人也聽」與「音色相近」的推薦。")
    with st.form("song_form"):
        sq = st.text_input("搜尋要作為起點的歌曲", placeholder="例如：clocks")
        go = st.form_submit_button("搜尋")
    if go and sq.strip():
        found, _ = search_songs(sq.strip(), 30)
        st.session_state["song_candidates"] = found

    cands = st.session_state.get("song_candidates", [])
    if cands:
        pick = st.selectbox("選擇歌曲", range(len(cands)), format_func=lambda i: label(cands[i]))
        seed = cands[pick]
        st.markdown(f"**起點歌曲：** {label(seed)}　｜　聽眾 {seed['listeners']:,}")

        left, right = st.columns(2)
        with left:
            st.subheader("👥 聽過的人也聽（物品協同過濾）")
            rows, err = get_item_cf(seed["song_id"], result_limit)
            if err:
                st.warning(err)
            else:
                show_songs(rows, "沒有足夠的共同聽眾資料")
        with right:
            st.subheader("🎛️ 音色相近（內容導向）")
            if not seed["has_audio_features"]:
                st.info("這首歌沒有對應的 Spotify 音訊特徵，無法做內容導向推薦。")
            else:
                rows, err = get_content(seed["song_id"], result_limit)
                if err:
                    st.warning(err)
                else:
                    show_songs(rows)
    else:
        st.info("請先搜尋歌曲。")

# --------------------------- 使用者推薦 ---------------------------
with tab_user:
    st.write("輸入使用者編號 (`user_idx`)，比較個人化推薦與熱門基準線。"
             "（只有通過 Chapter 4 過濾條件的使用者才有資料）")
    if "uid" not in st.session_state:
        st.session_state["uid"] = 0

    def _pick_random():
        st.session_state["uid"] = random_user()

    c1, c2 = st.columns([2, 1])
    uid = c1.number_input("使用者編號", min_value=0, step=1, key="uid")
    c2.write("")
    c2.button("🎲 隨機抽一位使用者", on_click=_pick_random)

    hist, herr = get_history(int(uid), result_limit)
    if herr:
        st.warning(herr)
    else:
        a, b, c = st.columns(3)
        with a:
            st.subheader("📜 聆聽紀錄（依評分）")
            show_songs(hist)
        with b:
            st.subheader("✨ 個人化推薦")
            rows, err = get_user_cf(int(uid), result_limit)
            if err:
                st.warning(err)
            else:
                show_songs(rows, "沒有找到推薦結果")
        with c:
            st.subheader("🔥 熱門基準線")
            rows, err = get_popular(result_limit, "listeners")
            show_songs(rows)

# ----------------------------- 概覽 -----------------------------
with tab_info:
    if st.button("計算資料統計"):
        st.session_state["overview"] = get_overview()
    if "overview" in st.session_state:
        info, years = st.session_state["overview"]
        cols = st.columns(len(info))
        for col, (k, v) in zip(cols, info.items()):
            col.metric(k, f"{v:,}" if isinstance(v, int) else v)
        st.subheader("各年份歌曲數")
        st.bar_chart(years.set_index("year")["n"])
    else:
        st.info("按下按鈕開始統計（資料量大，約需數秒）。")
