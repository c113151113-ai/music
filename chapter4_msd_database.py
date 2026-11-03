"""
Chapter 4: 以 Million Song Dataset 建立資料庫
  1. 建立 SQLite 資料庫與資料表結構
  2. 匯入 MSD 歌曲中繼資料 (unique_tracks.txt)
  3. 處理使用者聆聽紀錄 (Taste Profile: train_triplets.txt)
  4. 處理 Spotify 音樂資料表 (Kaggle CSV) 並與 MSD 歌曲比對
  5. 範例查詢

需要資料 (請先下載):
  - MSD Taste Profile:  train_triplets.txt      (user_id \t song_id \t play_count)
  - MSD 歌曲清單:        unique_tracks.txt       (track_id<SEP>song_id<SEP>artist<SEP>title)
  - Spotify 資料集:      spotify_tracks.csv      (Kaggle: Spotify Tracks Dataset 等)

執行範例:
  python chapter4_msd_database.py \
      --triplets data/train_triplets.txt \
      --tracks data/unique_tracks.txt \
      --spotify data/spotify_tracks.csv \
      --db msd.db
"""

import argparse
import os
import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. 建立資料庫結構
# --------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS songs (
    song_id     TEXT PRIMARY KEY,
    track_id    TEXT,
    artist_name TEXT,
    title       TEXT,
    release     TEXT,
    year        INTEGER,
    match_key   TEXT            -- 正規化後的 "artist|title"，用來和 Spotify 比對
);

CREATE TABLE IF NOT EXISTS users (
    user_idx    INTEGER PRIMARY KEY,   -- 整數編號，節省空間
    user_id     TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS listening_history (
    user_idx    INTEGER NOT NULL,
    song_id     TEXT NOT NULL,
    play_count  INTEGER NOT NULL,
    PRIMARY KEY (user_idx, song_id)
);

CREATE TABLE IF NOT EXISTS spotify_tracks (
    spotify_id        TEXT PRIMARY KEY,
    artist_name       TEXT,
    track_name        TEXT,
    album_name        TEXT,
    genre             TEXT,
    popularity        INTEGER,
    duration_ms       INTEGER,
    explicit          INTEGER,
    danceability      REAL,
    energy            REAL,
    loudness          REAL,
    speechiness       REAL,
    acousticness      REAL,
    instrumentalness  REAL,
    liveness          REAL,
    valence           REAL,
    tempo             REAL,
    match_key         TEXT,
    song_id           TEXT              -- 對應到 MSD 的 song_id (可能為 NULL)
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_songs_match     ON songs(match_key);
CREATE INDEX IF NOT EXISTS idx_hist_song       ON listening_history(song_id);
CREATE INDEX IF NOT EXISTS idx_spotify_match   ON spotify_tracks(match_key);
CREATE INDEX IF NOT EXISTS idx_spotify_song    ON spotify_tracks(song_id);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# 工具：文字正規化 (用於跨資料集比對歌名 / 歌手)
# --------------------------------------------------------------------------
def normalize_text(s) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)         # 移除括號內容 (feat., remastered…)
    s = re.sub(r"\s-\s.*$", " ", s)                # 移除 " - Remastered 2011" 之類尾巴
    s = re.sub(r"[^a-z0-9 ]", " ", s)              # 只保留英數
    return re.sub(r"\s+", " ", s).strip()


def make_key(artist, title) -> str:
    return f"{normalize_text(artist)}|{normalize_text(title)}"


# --------------------------------------------------------------------------
# 2. 匯入 MSD 歌曲中繼資料
# --------------------------------------------------------------------------
def load_songs(conn: sqlite3.Connection, tracks_path: str) -> None:
    print("[songs] 讀取 unique_tracks.txt ...")
    df = pd.read_csv(
        tracks_path,
        sep="<SEP>",
        engine="python",          # 多字元分隔符號需使用 python engine
        header=None,
        names=["track_id", "song_id", "artist_name", "title"],
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["song_id"])
    df = df.drop_duplicates(subset="song_id")          # 一首 song 可能對應多個 track
    df["match_key"] = [make_key(a, t) for a, t in zip(df["artist_name"], df["title"])]

    df.to_sql("songs", conn, if_exists="append", index=False,
              chunksize=10000)
    conn.commit()
    print(f"[songs] 匯入 {len(df):,} 首歌曲")


def load_songs_from_metadata_db(conn: sqlite3.Connection, meta_path: str) -> None:
    """從 MSD 提供的 track_metadata.db (SQLite) 匯入歌曲資料。"""
    print("[songs] 讀取 track_metadata.db ...")
    src = sqlite3.connect(meta_path)
    table = src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()[0]
    df = pd.read_sql(
        f"SELECT track_id, song_id, artist_name, title, release, year FROM {table}", src)
    src.close()

    df = df.dropna(subset=["song_id"])
    df = df.drop_duplicates(subset="song_id")
    df["year"] = df["year"].replace(0, np.nan).astype("Int64")   # year=0 代表未知
    df["match_key"] = [make_key(a, t) for a, t in zip(df["artist_name"], df["title"])]

    df.to_sql("songs", conn, if_exists="append", index=False,
              chunksize=10000)
    conn.commit()
    print(f"[songs] 匯入 {len(df):,} 首歌曲")


# --------------------------------------------------------------------------
# 3. 處理使用者聆聽紀錄 (Taste Profile)
# --------------------------------------------------------------------------
def load_listening_history(conn: sqlite3.Connection, triplets_path: str,
                           chunksize: int = 2_000_000) -> None:
    """
    train_triplets.txt 約 4800 萬筆，需分批讀取。
    - 以 user_id -> 整數 user_idx 對應，減少儲存空間
    - 只保留存在於 songs 表中的歌曲
    - 同一 (user, song) 重複紀錄則加總 play_count
    """
    print("[history] 讀取 train_triplets.txt (分批) ...")
    valid_songs = set(r[0] for r in conn.execute("SELECT song_id FROM songs"))

    user_map: dict[str, int] = {}
    total = 0

    reader = pd.read_csv(
        triplets_path,
        sep="\t",
        header=None,
        names=["user_id", "song_id", "play_count"],
        dtype={"user_id": str, "song_id": str, "play_count": "int32"},
        chunksize=chunksize,
    )

    for i, chunk in enumerate(reader, 1):
        chunk = chunk.dropna()
        chunk = chunk[chunk["song_id"].isin(valid_songs)]
        chunk = chunk[chunk["play_count"] > 0]

        # 新使用者寫入 users 表
        new_users = [u for u in chunk["user_id"].unique() if u not in user_map]
        if new_users:
            start = len(user_map)
            for j, u in enumerate(new_users):
                user_map[u] = start + j
            conn.executemany(
                "INSERT OR IGNORE INTO users(user_idx, user_id) VALUES (?, ?)",
                [(user_map[u], u) for u in new_users],
            )

        chunk["user_idx"] = chunk["user_id"].map(user_map).astype("int64")
        chunk = (chunk.groupby(["user_idx", "song_id"], as_index=False)["play_count"].sum())

        conn.executemany(
            "INSERT OR REPLACE INTO listening_history(user_idx, song_id, play_count) "
            "VALUES (?, ?, ?)",
            chunk.itertuples(index=False, name=None),
        )
        conn.commit()
        total += len(chunk)
        print(f"  chunk {i}: 累計 {total:,} 筆")

    print(f"[history] 完成，共 {total:,} 筆；使用者 {len(user_map):,} 人")


def _step(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filter_sparse_interactions(conn: sqlite3.Connection,
                               min_user_songs: int = 20,
                               min_song_users: int = 50) -> None:
    """
    推薦系統常見前處理：移除互動過少的使用者與歌曲 (降低稀疏度)。
    全程在 SQLite 內完成，不把幾千萬筆資料載入記憶體。
    產生資料表: good_users, good_songs, listening_filtered
    """
    _step(f"[filter] 保留 聆聽歌曲數>={min_user_songs} 的使用者、"
          f"被聆聽人數>={min_song_users} 的歌曲")
    conn.executescript("DROP TABLE IF EXISTS good_users; DROP TABLE IF EXISTS good_songs; "
                       "DROP TABLE IF EXISTS listening_filtered;")

    _step("[filter] 1/3 找出符合條件的使用者 ...")
    conn.execute(f"""CREATE TABLE good_users AS
        SELECT user_idx FROM listening_history
        GROUP BY user_idx HAVING COUNT(*) >= {min_user_songs}""")
    conn.execute("CREATE UNIQUE INDEX idx_gu ON good_users(user_idx)")

    _step("[filter] 2/3 找出符合條件的歌曲 ...")
    conn.execute(f"""CREATE TABLE good_songs AS
        SELECT song_id FROM listening_history
        GROUP BY song_id HAVING COUNT(*) >= {min_song_users}""")
    conn.execute("CREATE UNIQUE INDEX idx_gs ON good_songs(song_id)")
    conn.commit()

    _step("[filter] 3/3 篩選聆聽紀錄 ...")
    conn.execute("""CREATE TABLE listening_filtered AS
        SELECT h.user_idx, h.song_id, h.play_count
        FROM listening_history h
        JOIN good_users u ON u.user_idx = h.user_idx
        JOIN good_songs s ON s.song_id  = h.song_id""")
    conn.execute("CREATE INDEX idx_lf_user ON listening_filtered(user_idx)")
    conn.execute("CREATE INDEX idx_lf_song ON listening_filtered(song_id)")
    conn.commit()

    n, nu, ns = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT user_idx), COUNT(DISTINCT song_id) "
        "FROM listening_filtered").fetchone()
    _step(f"[filter] 剩 {n:,} 筆、{nu:,} 位使用者、{ns:,} 首歌")


def add_implicit_rating(conn: sqlite3.Connection) -> None:
    """
    播放次數 -> 隱式評分：log1p 壓縮長尾，再除以該使用者的最大值，正規化到 0~1。
    結果存成資料表 listening_rated (在 SQLite 內計算)。
    """
    import math
    conn.create_function("log1p", 1, math.log1p)
    _step("[rating] 計算隱式評分 ...")
    conn.executescript("DROP TABLE IF EXISTS user_max; DROP TABLE IF EXISTS listening_rated;")
    conn.execute("""CREATE TABLE user_max AS
        SELECT user_idx, MAX(log1p(play_count)) AS mx
        FROM listening_filtered GROUP BY user_idx""")
    conn.execute("CREATE UNIQUE INDEX idx_um ON user_max(user_idx)")
    conn.execute("""CREATE TABLE listening_rated AS
        SELECT f.user_idx, f.song_id, f.play_count,
               log1p(f.play_count) / m.mx AS rating
        FROM listening_filtered f JOIN user_max m ON m.user_idx = f.user_idx""")
    conn.execute("CREATE INDEX idx_lr_user ON listening_rated(user_idx)")
    conn.execute("CREATE INDEX idx_lr_song ON listening_rated(song_id)")
    conn.commit()
    _step("[rating] 完成，結果在資料表 listening_rated")


def describe_history(conn: sqlite3.Connection) -> None:
    n_users, n_songs, n_rows = conn.execute(
        "SELECT COUNT(DISTINCT user_idx), COUNT(DISTINCT song_id), COUNT(*) "
        "FROM listening_history"
    ).fetchone()
    sparsity = 1 - n_rows / (n_users * n_songs)
    print(f"[stats] 使用者 {n_users:,} | 歌曲 {n_songs:,} | 紀錄 {n_rows:,} | "
          f"稀疏度 {sparsity:.6f}")


# --------------------------------------------------------------------------
# 4. 處理 Spotify 音樂資料表
# --------------------------------------------------------------------------
# 不同版本的 Kaggle CSV 欄位名稱不同，這裡統一對應
COLUMN_ALIASES = {
    "spotify_id":       ["track_id", "id", "spotify_id", "uri"],
    "artist_name":      ["artists", "artist_name", "artist"],
    "track_name":       ["track_name", "name", "title"],
    "album_name":       ["album_name", "album"],
    "genre":            ["track_genre", "genre", "genres"],
    "popularity":       ["popularity"],
    "duration_ms":      ["duration_ms", "duration"],
    "explicit":         ["explicit"],
    "danceability":     ["danceability"],
    "energy":           ["energy"],
    "loudness":         ["loudness"],
    "speechiness":      ["speechiness"],
    "acousticness":     ["acousticness"],
    "instrumentalness": ["instrumentalness"],
    "liveness":         ["liveness"],
    "valence":          ["valence"],
    "tempo":            ["tempo"],
}

AUDIO_FEATURES = ["danceability", "energy", "loudness", "speechiness",
                  "acousticness", "instrumentalness", "liveness", "valence", "tempo"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    lower = {c.lower(): c for c in df.columns}
    for target, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[target] = df[lower[a]]
                break
        else:
            out[target] = np.nan        # 缺少的欄位補空值
    return out


def clean_spotify(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    # 缺值與重複
    df = df.dropna(subset=["spotify_id", "artist_name", "track_name"])
    df = df.drop_duplicates(subset="spotify_id")

    # 型別與合理範圍
    df["explicit"] = df["explicit"].map(
        {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    for col in ["popularity", "duration_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in AUDIO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["duration_ms"].between(30_000, 1_800_000) | df["duration_ms"].isna()]
    df = df[(df["tempo"] > 0) | df["tempo"].isna()]

    # 音訊特徵缺值以中位數補齊
    for col in AUDIO_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # 多位歌手 (以 ; 或 , 分隔) 取第一位作為比對用主要歌手
    primary_artist = df["artist_name"].astype(str).str.split(r";|,", regex=True).str[0]
    df["match_key"] = [make_key(a, t) for a, t in zip(primary_artist, df["track_name"])]

    # 同一 match_key 保留人氣最高者，避免 1 對多比對
    df = df.sort_values("popularity", ascending=False)
    return df


def load_spotify(conn: sqlite3.Connection, spotify_path: str) -> None:
    print("[spotify] 讀取 CSV ...")
    raw = pd.read_csv(spotify_path, low_memory=False)
    print(f"[spotify] 原始 {len(raw):,} 筆")
    df = clean_spotify(raw)
    df["song_id"] = None
    cols = ["spotify_id", "artist_name", "track_name", "album_name", "genre",
            "popularity", "duration_ms", "explicit", *AUDIO_FEATURES,
            "match_key", "song_id"]
    df[cols].to_sql("spotify_tracks", conn, if_exists="append", index=False,
                    chunksize=10000)
    conn.commit()
    print(f"[spotify] 清理後匯入 {len(df):,} 筆")


def link_spotify_to_msd(conn: sqlite3.Connection) -> None:
    """以正規化後的 artist|title 將 Spotify 曲目對應到 MSD song_id。"""
    print("[link] 比對 Spotify 與 MSD ...")
    conn.executescript("""
        UPDATE spotify_tracks
        SET song_id = (
            SELECT s.song_id FROM songs s
            WHERE s.match_key = spotify_tracks.match_key
              AND s.match_key <> '|'
            LIMIT 1
        )
        WHERE match_key <> '|';
    """)
    conn.commit()
    matched, total = conn.execute(
        "SELECT SUM(song_id IS NOT NULL), COUNT(*) FROM spotify_tracks").fetchone()
    print(f"[link] 成功對應 {matched or 0:,} / {total:,} "
          f"({(matched or 0) / max(total, 1):.1%})")


# --------------------------------------------------------------------------
# 5. 範例查詢
# --------------------------------------------------------------------------
def demo_queries(conn: sqlite3.Connection) -> None:
    print("\n=== 最熱門歌曲 (依總播放次數) ===")
    q1 = """
        SELECT s.artist_name, s.title,
               SUM(h.play_count) AS total_plays,
               COUNT(*)          AS listeners
        FROM listening_history h JOIN songs s USING (song_id)
        GROUP BY h.song_id
        ORDER BY total_plays DESC
        LIMIT 10;
    """
    print(pd.read_sql(q1, conn).to_string(index=False))

    print("\n=== 已對應 Spotify 的熱門歌曲及其音訊特徵 ===")
    q2 = """
        SELECT s.artist_name, s.title,
               SUM(h.play_count) AS total_plays,
               sp.danceability, sp.energy, sp.valence, sp.tempo
        FROM listening_history h
        JOIN songs s           ON s.song_id = h.song_id
        JOIN spotify_tracks sp ON sp.song_id = h.song_id
        GROUP BY h.song_id
        ORDER BY total_plays DESC
        LIMIT 10;
    """
    print(pd.read_sql(q2, conn).to_string(index=False))

    print("\n=== 某位使用者的聆聽紀錄 (user_idx = 0) ===")
    q3 = """
        SELECT s.artist_name, s.title, h.play_count
        FROM listening_history h JOIN songs s USING (song_id)
        WHERE h.user_idx = 0
        ORDER BY h.play_count DESC
        LIMIT 10;
    """
    print(pd.read_sql(q3, conn).to_string(index=False))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Chapter 4: MSD 資料庫建置")
    base = os.path.dirname(os.path.abspath(__file__))   # 程式所在資料夾 (不受終端機位置影響)
    ap.add_argument("--triplets", default=os.path.join(base, "train_triplets.txt"),
                    help="train_triplets.txt 路徑 (預設: 與程式同資料夾)")
    ap.add_argument("--metadata-db", default=os.path.join(base, "track_metadata.db"),
                    help="MSD track_metadata.db 路徑 (SQLite)")
    ap.add_argument("--tracks", default=None,
                    help="unique_tracks.txt 路徑 (若沒有 track_metadata.db 才用)")
    ap.add_argument("--spotify", default=os.path.join(base, "dataset.csv"),
                    help="Spotify CSV 路徑 (預設: dataset.csv)")
    ap.add_argument("--db", default=os.path.join(base, "msd.db"), help="輸出 SQLite 檔名")
    ap.add_argument("--reset", action="store_true", help="若輸出資料庫已存在則刪除重建")
    ap.add_argument("--resume", action="store_true",
                    help="沿用已匯入完成的資料庫，跳過匯入，只做 建索引 -> 比對 -> 過濾")
    ap.add_argument("--export-csv", action="store_true",
                    help="額外把評分結果匯出成 CSV (預設只存在資料庫的 listening_rated 資料表)")
    ap.add_argument("--min-user-songs", type=int, default=20)
    ap.add_argument("--min-song-users", type=int, default=50)
    args = ap.parse_args()

    # 先確認輸入檔存在 (sqlite3.connect 對不存在的檔案會悄悄建立一個空檔)
    required = [] if args.resume else [args.triplets, args.tracks or args.metadata_db]
    for path in required:
        if not os.path.isfile(path):
            raise SystemExit(f"找不到檔案: {path}\n請確認檔名與位置，或用參數指定路徑。")

    if args.resume:
        if not os.path.isfile(args.db):
            raise SystemExit(f"找不到資料庫: {args.db}，無法 --resume。")
    elif os.path.exists(args.db):
        if args.reset:
            os.remove(args.db)
        else:
            raise SystemExit(f"{args.db} 已存在。請加上 --reset 重建，或用 --db 指定新檔名。")

    conn = init_db(args.db)

    if not args.resume:
        if args.tracks:
            load_songs(conn, args.tracks)
        else:
            load_songs_from_metadata_db(conn, args.metadata_db)
        load_listening_history(conn, args.triplets)
        describe_history(conn)

        if args.spotify and os.path.exists(args.spotify):
            load_spotify(conn, args.spotify)

    # 重要：必須「先建索引、再比對」，否則比對時每一列都會掃描整張 songs 表，慢到無法完成
    print("[index] 建立索引 ...")
    conn.executescript(INDEXES)
    conn.commit()

    if conn.execute("SELECT COUNT(*) FROM spotify_tracks").fetchone()[0] > 0:
        link_spotify_to_msd(conn)

    conn.execute("PRAGMA cache_size = -200000")      # 約 200MB 快取
    filter_sparse_interactions(conn, args.min_user_songs, args.min_song_users)
    add_implicit_rating(conn)

    if args.export_csv:
        _step("[export] 匯出 listening_rated.csv (檔案可能很大) ...")
        out = os.path.join(os.path.dirname(os.path.abspath(args.db)), "listening_rated.csv")
        first = True
        for chunk in pd.read_sql("SELECT * FROM listening_rated", conn, chunksize=2_000_000):
            chunk.to_csv(out, mode="w" if first else "a", header=first, index=False)
            first = False
        _step(f"[export] 完成: {out}")

    demo_queries(conn)
    conn.close()
    print("\n完成！資料庫:", args.db)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
