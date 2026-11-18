"""
modules/metadata_manager.py
SQLite schema management, JSON metadata export, and social media performance tracking.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import sqlite3
import json
from datetime import datetime

# ─────────────────────── SCHEMAS ───────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id          TEXT PRIMARY KEY,
    source_path      TEXT,
    source_url       TEXT,
    start_sec        REAL,
    end_sec          REAL,
    duration         REAL,
    category         TEXT,
    score            REAL,
    hook             TEXT,
    alt_hooks        TEXT,
    selected_title   TEXT,
    alt_titles       TEXT,
    yt_description   TEXT,
    ig_description   TEXT,
    tt_description   TEXT,
    hashtags         TEXT,
    thumbnail_path   TEXT,
    clip_path        TEXT,
    qc_passed        INTEGER DEFAULT 0,
    qc_issues        TEXT,
    export_settings  TEXT,
    created_at       TEXT
);
"""

CREATE_PERFORMANCE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clip_performance (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id          TEXT NOT NULL,
    platform         TEXT NOT NULL,
    views            INTEGER DEFAULT 0,
    likes            INTEGER DEFAULT 0,
    shares           INTEGER DEFAULT 0,
    comments         INTEGER DEFAULT 0,
    avg_watch_sec    REAL DEFAULT 0.0,
    completion_rate  REAL DEFAULT 0.0,
    subs_gained      INTEGER DEFAULT 0,
    updated_at       TEXT,
    FOREIGN KEY(clip_id) REFERENCES clips(clip_id)
);
"""

CREATE_LEGACY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clips_legacy (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_path        TEXT,
    start_sec        REAL,
    end_sec          REAL,
    title            TEXT,
    description      TEXT,
    score            REAL,
    created_at       TEXT
);
"""


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialise SQLite database with clips and performance tables."""
    db_path = db_path or config.DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(CREATE_TABLE_SQL + CREATE_PERFORMANCE_TABLE_SQL + CREATE_LEGACY_TABLE_SQL)
    conn.commit()
    return conn


def save_clip_metadata(clip_data: dict, db_path: str | None = None) -> str:
    """Insert or replace a clip record in the database."""
    db_path = db_path or config.DB_PATH
    conn = init_db(db_path)

    clip_id = clip_data.get("clip_id") or _generate_clip_id(clip_data)
    now     = datetime.now().isoformat()

    def _j(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return val

    conn.execute(
        """INSERT OR REPLACE INTO clips VALUES (
            :clip_id, :source_path, :source_url,
            :start_sec, :end_sec, :duration,
            :category, :score,
            :hook, :alt_hooks, :selected_title, :alt_titles,
            :yt_description, :ig_description, :tt_description, :hashtags,
            :thumbnail_path, :clip_path,
            :qc_passed, :qc_issues, :export_settings, :created_at
        )""",
        {
            "clip_id":        clip_id,
            "source_path":    clip_data.get("source_path", ""),
            "source_url":     clip_data.get("source_url", ""),
            "start_sec":      clip_data.get("start_sec", 0.0),
            "end_sec":        clip_data.get("end_sec", 0.0),
            "duration":       clip_data.get("duration", 0.0),
            "category":       clip_data.get("category", "unknown"),
            "score":          clip_data.get("score", 0.0),
            "hook":           clip_data.get("hook", ""),
            "alt_hooks":      _j(clip_data.get("alt_hooks", [])),
            "selected_title": clip_data.get("selected_title", ""),
            "alt_titles":     _j(clip_data.get("alt_titles", [])),
            "yt_description": clip_data.get("yt_description", ""),
            "ig_description": clip_data.get("ig_description", ""),
            "tt_description": clip_data.get("tt_description", ""),
            "hashtags":       _j(clip_data.get("hashtags", [])),
            "thumbnail_path": clip_data.get("thumbnail_path", ""),
            "clip_path":      clip_data.get("clip_path", ""),
            "qc_passed":      int(clip_data.get("qc_passed", False)),
            "qc_issues":      _j(clip_data.get("qc_issues", [])),
            "export_settings":_j(clip_data.get("export_settings", {})),
            "created_at":     now,
        }
    )
    conn.commit()
    conn.close()
    return clip_id


def save_performance_metrics(clip_id: str, platform: str, metrics: dict, db_path: str | None = None):
    """Save manual performance metrics for a clip on a platform."""
    db_path = db_path or config.DB_PATH
    conn = init_db(db_path)
    now = datetime.now().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO clip_performance (
            clip_id, platform, views, likes, shares, comments, avg_watch_sec, completion_rate, subs_gained, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clip_id,
            platform,
            metrics.get("views", 0),
            metrics.get("likes", 0),
            metrics.get("shares", 0),
            metrics.get("comments", 0),
            metrics.get("avg_watch_sec", 0.0),
            metrics.get("completion_rate", 0.0),
            metrics.get("subs_gained", 0),
            now,
        )
    )
    conn.commit()
    conn.close()


def get_all_clips(db_path: str | None = None) -> list[dict]:
    """Return all clip records as a list of dicts."""
    db_path = db_path or config.DB_PATH
    conn = init_db(db_path)
    rows = conn.execute("SELECT * FROM clips ORDER BY created_at DESC").fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        for key in ("alt_hooks", "alt_titles", "hashtags", "qc_issues", "export_settings"):
            try:
                d[key] = json.loads(d[key]) if d[key] else []
            except Exception:
                pass
        result.append(d)
    return result


def export_clip_metadata_json(clip_id: str, output_dir: str, db_path: str | None = None) -> str | None:
    """Write clip metadata to {output_dir}/metadata.json."""
    db_path = db_path or config.DB_PATH
    conn = init_db(db_path)
    row = conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
    conn.close()
    if row is None:
        return None

    data = dict(row)
    for key in ("alt_hooks", "alt_titles", "hashtags", "qc_issues", "export_settings"):
        try:
            data[key] = json.loads(data[key]) if data[key] else []
        except Exception:
            pass

    out = os.path.join(output_dir, "metadata.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


def _generate_clip_id(clip_data: dict) -> str:
    import hashlib
    src   = str(clip_data.get("source_path", ""))
    start = str(clip_data.get("start_sec", 0))
    end   = str(clip_data.get("end_sec", 0))
    ts    = datetime.now().isoformat()
    raw   = f"{src}-{start}-{end}-{ts}"
    return "clip_" + hashlib.md5(raw.encode()).hexdigest()[:10]
