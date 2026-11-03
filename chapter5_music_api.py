"""
Chapter 5: 以 FastAPI 實作音樂搜尋和推薦演算法

建立在 Chapter 4 產生的 msd.db 之上，提供：
  搜尋        GET /search?q=...                      (SQLite FTS5 全文檢索，依熱門度排序)
  熱門推薦    GET /recommend/popular                 (非個人化基準線)
  物品協同    GET /recommend/item/{song_id}          (聽過這首歌的人也聽了…)
  使用者協同  GET /recommend/user/{user_idx}         (找相似使用者，推薦他們聽的歌)
  內容導向    GET /recommend/content/{song_id}       (以 Spotify 音訊特徵找音色相近的歌)
  其他        GET /songs/{song_id}、GET /users/{user_idx}/history、首頁 /、互動文件 /docs

安裝套件:
  pip install fastapi uvicorn numpy pandas

使用步驟:
  1) python chapter5_music_api.py --prepare     # 只需做一次：建立 song_stats 與全文檢索索引
  2) python chapter5_music_api.py               # 啟動服務
  3) 瀏覽器開啟 http://127.0.0.1:8000  (測試介面) 或 http://127.0.0.1:8000/docs (API 文件)

資料庫路徑預設為與本程式同資料夾的 msd.db，可用環境變數 MSD_DB 覆蓋。
"""

import argparse
import heapq
import math
import os
import re
import sqlite3
from collections import defaultdict
from contextlib import asynccontextmanager
from operator import itemgetter
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("MSD_DB", os.path.join(BASE, "msd.db"))

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]


# --------------------------------------------------------------------------
# 一次性準備：統計表 + 全文檢索索引
# --------------------------------------------------------------------------
def prepare_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -200000")

    print("[prepare] 建立 song_stats (每首歌的總播放數與聽眾數) ...", flush=True)
    conn.execute("DROP TABLE IF EXISTS song_stats")
    conn.execute("""
        CREATE TABLE song_stats AS
        SELECT song_id, SUM(play_count) AS total_plays, COUNT(*) AS listeners
        FROM listening_filtered GROUP BY song_id""")
    conn.execute("CREATE UNIQUE INDEX idx_ss ON song_stats(song_id)")
    conn.commit()

    print("[prepare] 建立全文檢索索引 songs_fts ...", flush=True)
    try:
        conn.execute("DROP TABLE IF EXISTS songs_fts")
        conn.execute("""CREATE VIRTUAL TABLE songs_fts USING fts5(
                            title, artist_name, content='songs', content_rowid='rowid')""")
        conn.execute("INSERT INTO songs_fts(rowid, title, artist_name) "
                     "SELECT rowid, title, artist_name FROM songs")
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"[prepare] 此 SQLite 不支援 FTS5 ({e})，搜尋將改用 LIKE (較慢)。")
    conn.close()
    print("[prepare] 完成。現在可以啟動服務：python chapter5_music_api.py")


# --------------------------------------------------------------------------
# 資料模型 / 工具
# --------------------------------------------------------------------------
class SongOut(BaseModel):
    song_id: str
    title: Optional[str] = None
    artist_name: Optional[str] = None
    year: Optional[int] = None
    total_plays: int = 0
    listeners: int = 0
    has_audio_features: bool = False
    score: Optional[float] = None


SONG_COLS = ("s.song_id, s.title, s.artist_name, s.year, "
             "COALESCE(st.total_plays, 0), COALESCE(st.listeners, 0)")
SONG_FROM = "songs s LEFT JOIN song_stats st ON st.song_id = s.song_id"

STATE: dict = {}      # 內容導向推薦用的特徵矩陣 (啟動時載入)


def to_song(row, score: Optional[float] = None) -> SongOut:
    song_id, title, artist, year, plays, listeners = row
    return SongOut(
        song_id=song_id, title=title, artist_name=artist,
        year=int(year) if year else None,
        total_plays=int(plays), listeners=int(listeners),
        has_audio_features=song_id in STATE.get("pos", {}),
        score=None if score is None else round(float(score), 4),
    )


def songs_by_ids(conn, ids: list[str]) -> dict:
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT {SONG_COLS} FROM {SONG_FROM} WHERE s.song_id IN ({q})", ids)
    return {r[0]: r for r in rows}


def ranked_songs(conn, scored: list[tuple[str, float]]) -> list[SongOut]:
    """scored = [(song_id, score), ...] 已排序 -> SongOut 清單。"""
    info = songs_by_ids(conn, [sid for sid, _ in scored])
    return [to_song(info[sid], sc) for sid, sc in scored if sid in info]


def has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def fts_query(q: str) -> str:
    tokens = re.findall(r"\w+", q)
    return " ".join(f'"{t}"*' for t in tokens)          # 每個字詞做前綴比對，彼此 AND


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA cache_size = -100000")
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 推薦演算法核心 (與 FastAPI 無關，可單獨測試)
# --------------------------------------------------------------------------
def popular_songs(conn, limit: int, by: str = "listeners") -> list[SongOut]:
    col = "st.total_plays" if by == "plays" else "st.listeners"
    rows = conn.execute(
        f"SELECT {SONG_COLS} FROM song_stats st JOIN songs s ON s.song_id = st.song_id "
        f"ORDER BY {col} DESC LIMIT ?", (limit,)).fetchall()
    return [to_song(r, r[5] if by == "listeners" else r[4]) for r in rows]


def item_cf(conn, song_id: str, limit: int,
            sample_users: int = 3000, min_co: int = 3) -> list[tuple[str, float]]:
    """
    物品協同過濾 (item-based CF, 以共同聽眾計算餘弦相似度)。
      相似度(x, y) = |聽過 x 且聽過 y 的人| / sqrt(|聽 x 的人| * |聽 y 的人|)
    為了即時回應，只抽樣最多 sample_users 位聽過 x 的使用者，再依抽樣比例還原共同聽眾數。
    """
    row = conn.execute("SELECT listeners FROM song_stats WHERE song_id = ?", (song_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "此歌曲不在過濾後的聆聽資料中，無法做協同過濾")
    n_x = row[0]
    s = min(sample_users, n_x)

    rows = conn.execute("""
        WITH u AS (SELECT user_idx FROM listening_filtered WHERE song_id = ? LIMIT ?)
        SELECT f.song_id, COUNT(*) AS co, st.listeners
        FROM listening_filtered f
        JOIN u ON f.user_idx = u.user_idx
        JOIN song_stats st ON st.song_id = f.song_id
        WHERE f.song_id <> ?
        GROUP BY f.song_id HAVING COUNT(*) >= ?""",
        (song_id, sample_users, song_id, min_co)).fetchall()

    # co_full ≈ co * n_x / s  =>  cosine ≈ (co / s) * sqrt(n_x / n_y)
    scores = {sid: (co / s) * math.sqrt(n_x / n_y) for sid, co, n_y in rows}
    return heapq.nlargest(limit, scores.items(), key=itemgetter(1))


def user_cf(conn, user_idx: int, limit: int,
            n_neighbors: int = 50, alpha: float = 0.3) -> list[tuple[str, float]]:
    """
    使用者協同過濾 (user-based CF)：
      1. 取該使用者評分最高的歌 (排除過於熱門、資訊量低的歌)
      2. 找出和他重疊歌曲最多的 n_neighbors 位鄰居
      3. 鄰居聽過、但他沒聽過的歌，以 (重疊數 × 鄰居評分) 加權累計
      4. 除以 listeners**alpha 抑制熱門歌曲過度主導 (alpha 越大越偏向冷門歌)
    """
    mine = conn.execute(
        "SELECT song_id, rating FROM listening_rated WHERE user_idx = ? ORDER BY rating DESC",
        (user_idx,)).fetchall()
    if not mine:
        raise HTTPException(404, "找不到此使用者 (可能已被過濾條件排除)")
    heard = {sid for sid, _ in mine}

    top = [sid for sid, _ in mine[:30]]
    q = ",".join("?" * len(top))
    pop = dict(conn.execute(f"SELECT song_id, listeners FROM song_stats WHERE song_id IN ({q})", top))
    seeds = [sid for sid in top if pop.get(sid, 0) <= 20000] or top[:5]

    q = ",".join("?" * len(seeds))
    neighbors = conn.execute(
        f"SELECT user_idx, COUNT(*) AS overlap FROM listening_filtered "
        f"WHERE song_id IN ({q}) AND user_idx <> ? "
        f"GROUP BY user_idx ORDER BY overlap DESC LIMIT ?",
        (*seeds, user_idx, n_neighbors)).fetchall()
    if not neighbors:
        return []
    weight = dict(neighbors)

    q = ",".join("?" * len(weight))
    cand = conn.execute(
        f"SELECT r.user_idx, r.song_id, r.rating, st.listeners "
        f"FROM listening_rated r JOIN song_stats st ON st.song_id = r.song_id "
        f"WHERE r.user_idx IN ({q})", tuple(weight))
    scores: dict[str, float] = defaultdict(float)
    for u, sid, rating, n in cand:
        if sid not in heard:
            scores[sid] += weight[u] * rating / (n ** alpha)
    return heapq.nlargest(limit, scores.items(), key=itemgetter(1))


def content_similar(song_id: str, limit: int) -> list[tuple[str, float]]:
    """內容導向：以標準化後的 Spotify 音訊特徵計算餘弦相似度。"""
    pos = STATE.get("pos", {})
    if song_id not in pos:
        raise HTTPException(404, "此歌曲沒有對應的 Spotify 音訊特徵 (搜尋結果中 has_audio_features=true 才可用)")
    m = STATE["matrix"]
    sims = m @ m[pos[song_id]]
    order = np.argsort(-sims)
    ids = STATE["ids"]
    out = []
    for i in order:
        if ids[i] != song_id:
            out.append((ids[i], float(sims[i])))
            if len(out) >= limit:
                break
    return out


def load_content_matrix() -> None:
    conn = sqlite3.connect(DB_PATH)
    cols = ", ".join(AUDIO_FEATURES)
    rows = conn.execute(
        f"SELECT song_id, {cols} FROM spotify_tracks WHERE song_id IS NOT NULL "
        f"ORDER BY popularity DESC").fetchall()
    conn.close()
    seen, ids, feats = set(), [], []
    for r in rows:                       # 同一首 MSD 歌可能對到多筆 Spotify，只留人氣最高的
        if r[0] not in seen:
            seen.add(r[0])
            ids.append(r[0])
            feats.append(r[1:])
    if not ids:
        STATE.update(ids=[], pos={}, matrix=np.zeros((0, len(AUDIO_FEATURES)), dtype=np.float32))
        return
    x = np.asarray(feats, dtype=np.float32)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)             # z-score 標準化
    x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)         # 單位向量 -> 內積即餘弦
    STATE.update(ids=ids, pos={s: i for i, s in enumerate(ids)}, matrix=x)


# --------------------------------------------------------------------------
# FastAPI 應用
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isfile(DB_PATH):
        raise RuntimeError(f"找不到資料庫 {DB_PATH}，請先完成 Chapter 4。")
    conn = sqlite3.connect(DB_PATH)
    missing = [t for t in ("song_stats", "listening_filtered", "listening_rated")
               if not has_table(conn, t)]
    conn.close()
    if missing:
        raise RuntimeError(f"資料庫缺少 {missing}。請先執行: python chapter5_music_api.py --prepare")
    load_content_matrix()
    print(f"[startup] 音訊特徵矩陣: {len(STATE['ids']):,} 首歌")
    yield


app = FastAPI(title="音樂搜尋與推薦 API (Million Song Dataset)", lifespan=lifespan)


@app.get("/search", response_model=list[SongOut], summary="搜尋歌曲 (歌名/歌手)")
def search(q: str = Query(..., min_length=1, description="歌名或歌手關鍵字"),
           limit: int = Query(20, ge=1, le=100), conn=Depends(get_conn)):
    match = fts_query(q)
    if not match:
        return []
    if has_table(conn, "songs_fts"):
        rows = conn.execute(
            f"SELECT {SONG_COLS} FROM songs_fts JOIN songs s ON s.rowid = songs_fts.rowid "
            f"LEFT JOIN song_stats st ON st.song_id = s.song_id "
            f"WHERE songs_fts MATCH ? ORDER BY COALESCE(st.listeners, 0) DESC LIMIT ?",
            (match, limit)).fetchall()
    else:
        like = f"%{q}%"
        rows = conn.execute(
            f"SELECT {SONG_COLS} FROM {SONG_FROM} WHERE s.title LIKE ? OR s.artist_name LIKE ? "
            f"ORDER BY COALESCE(st.listeners, 0) DESC LIMIT ?", (like, like, limit)).fetchall()
    return [to_song(r) for r in rows]


@app.get("/songs/{song_id}", response_model=SongOut, summary="單一歌曲資訊")
def get_song(song_id: str, conn=Depends(get_conn)):
    info = songs_by_ids(conn, [song_id])
    if song_id not in info:
        raise HTTPException(404, "找不到歌曲")
    return to_song(info[song_id])


@app.get("/users/{user_idx}/history", response_model=list[SongOut], summary="使用者聆聽紀錄 (依評分)")
def user_history(user_idx: int, limit: int = Query(20, ge=1, le=100), conn=Depends(get_conn)):
    rows = conn.execute(
        "SELECT song_id, rating FROM listening_rated WHERE user_idx = ? "
        "ORDER BY rating DESC LIMIT ?", (user_idx, limit)).fetchall()
    if not rows:
        raise HTTPException(404, "找不到此使用者")
    return ranked_songs(conn, rows)


@app.get("/recommend/popular", response_model=list[SongOut], summary="熱門歌曲 (非個人化基準線)")
def rec_popular(limit: int = Query(10, ge=1, le=100),
                by: str = Query("listeners", pattern="^(listeners|plays)$"),
                conn=Depends(get_conn)):
    return popular_songs(conn, limit, by)


@app.get("/recommend/item/{song_id}", response_model=list[SongOut],
         summary="物品協同過濾：喜歡這首歌的人也聽…")
def rec_item(song_id: str, limit: int = Query(10, ge=1, le=100), conn=Depends(get_conn)):
    return ranked_songs(conn, item_cf(conn, song_id, limit))


@app.get("/recommend/user/{user_idx}", response_model=list[SongOut],
         summary="使用者協同過濾：為使用者個人化推薦")
def rec_user(user_idx: int, limit: int = Query(10, ge=1, le=100), conn=Depends(get_conn)):
    return ranked_songs(conn, user_cf(conn, user_idx, limit))


@app.get("/recommend/content/{song_id}", response_model=list[SongOut],
         summary="內容導向：音訊特徵相近的歌 (限已對應 Spotify 的歌曲)")
def rec_content(song_id: str, limit: int = Query(10, ge=1, le=100), conn=Depends(get_conn)):
    return ranked_songs(conn, content_similar(song_id, limit))


INDEX_HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>音樂搜尋與推薦</title>
<style>
 body{font-family:sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem}
 input{padding:.4rem} button{padding:.35rem .7rem;margin:.1rem;cursor:pointer}
 li{margin:.5rem 0} .meta{color:#666;font-size:.85rem}
</style></head><body>
<h2>音樂搜尋與推薦 (Million Song Dataset)</h2>
<div><input id="q" size="30" placeholder="歌名或歌手，例如 yellow">
 <button onclick="doSearch()">搜尋</button> <button onclick="popular()">熱門歌曲</button></div>
<div style="margin-top:.6rem">使用者編號 <input id="u" size="8" value="0">
 <button onclick="userRec()">為使用者推薦</button>
 <button onclick="userHist()">聆聽紀錄</button></div>
<h3 id="title"></h3><ul id="list"></ul>
<script>
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
async function show(title, url) {
  const r = await fetch(url), d = await r.json();
  document.getElementById('title').textContent = title;
  const ul = document.getElementById('list'); ul.innerHTML = '';
  if (!r.ok) { ul.innerHTML = '<li></li>'; ul.firstChild.textContent = d.detail || '發生錯誤'; return; }
  d.forEach(s => {
    const li = document.createElement('li');
    li.textContent = `${s.artist_name} — ${s.title}` + (s.year ? ` (${s.year})` : '');
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = `聽眾 ${s.listeners.toLocaleString()} | 播放 ${s.total_plays.toLocaleString()}` +
      (s.score != null ? ` | 分數 ${s.score}` : '');
    li.appendChild(meta);
    const b1 = document.createElement('button'); b1.textContent = '大家也聽 (協同過濾)';
    b1.onclick = () => show(`聽過「${s.title}」的人也聽`, '/recommend/item/' + s.song_id);
    li.appendChild(b1);
    if (s.has_audio_features) {
      const b2 = document.createElement('button'); b2.textContent = '音色相近 (內容導向)';
      b2.onclick = () => show(`與「${s.title}」音色相近`, '/recommend/content/' + s.song_id);
      li.appendChild(b2);
    }
    ul.appendChild(li);
  });
}
const doSearch = () => show('搜尋結果', '/search?q=' + encodeURIComponent(document.getElementById('q').value));
const popular = () => show('熱門歌曲', '/recommend/popular?limit=20');
const userRec = () => show('為使用者推薦', '/recommend/user/' + document.getElementById('u').value);
const userHist = () => show('使用者聆聽紀錄', '/users/' + document.getElementById('u').value + '/history');
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX_HTML


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chapter 5: FastAPI 音樂搜尋與推薦")
    ap.add_argument("--prepare", action="store_true", help="建立 song_stats 與全文檢索索引 (只需執行一次)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.prepare:
        prepare_db()
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
