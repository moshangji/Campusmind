import os
import logging
import importlib.util
from contextlib import contextmanager
from typing import Optional, List, Tuple

try:
    import pymysql  # type: ignore
except ImportError:  # pragma: no cover
    pymysql = None  # type: ignore

logger = logging.getLogger(__name__)

_TABLES_INITED = False

_CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      username VARCHAR(64) UNIQUE NOT NULL,
      password_hash VARCHAR(256) NOT NULL,
      role VARCHAR(32) NOT NULL DEFAULT 'student',
      display_name VARCHAR(64) DEFAULT '',
      school_id VARCHAR(64) DEFAULT '',
      identity_tag VARCHAR(32) DEFAULT '',
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      user_id BIGINT NULL,
      user_identity VARCHAR(32) NULL,
      school_id VARCHAR(64) NOT NULL DEFAULT '',
      created_at DATETIME NOT NULL,
      user_text TEXT NOT NULL,
      mood_state VARCHAR(64) NULL,
      top_emotion VARCHAR(64) NULL,
      probability FLOAT NULL,
      risk_level VARCHAR(32) NULL,
      intervention TEXT NULL
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS questionnaires (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      title VARCHAR(128) NOT NULL,
      description TEXT,
      scoring_rules JSON,
      score_multiplier FLOAT DEFAULT 1.0,
      is_active BOOLEAN DEFAULT TRUE,
      school_id VARCHAR(64) NOT NULL DEFAULT '',
      created_by BIGINT NULL,
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS questionnaire_items (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      questionnaire_id BIGINT NOT NULL,
      item_index INT NOT NULL,
      question_text TEXT NOT NULL,
      options JSON NOT NULL,
      is_reverse BOOLEAN DEFAULT FALSE
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS questionnaire_results (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      user_id BIGINT NULL,
      questionnaire_id BIGINT NOT NULL,
      answers JSON NOT NULL,
      raw_score INT DEFAULT 0,
      standard_score FLOAT DEFAULT 0,
      result_level VARCHAR(64) DEFAULT '',
      school_id VARCHAR(64) NOT NULL DEFAULT '',
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_lexicon (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      phrase VARCHAR(512) NOT NULL,
      category VARCHAR(32) NOT NULL,
      is_active BOOLEAN DEFAULT TRUE,
      sort_order INT DEFAULT 0,
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_risk_lex_cat (category, is_active)
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_config (
      config_key VARCHAR(64) PRIMARY KEY,
      config_value VARCHAR(256) NOT NULL,
      updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """,
]

_ADD_COLUMNS_SQL = [
    "ALTER TABLE conversations ADD COLUMN user_id BIGINT NULL",
    "ALTER TABLE conversations ADD COLUMN school_id VARCHAR(64) NOT NULL DEFAULT ''",
    "ALTER TABLE conversations ADD COLUMN mood_state VARCHAR(64) NULL",
    "ALTER TABLE questionnaires ADD COLUMN school_id VARCHAR(64) NOT NULL DEFAULT ''",
    "ALTER TABLE questionnaire_results ADD COLUMN school_id VARCHAR(64) NOT NULL DEFAULT ''",
    "ALTER TABLE users MODIFY COLUMN role VARCHAR(32) NOT NULL DEFAULT 'student'",
    "ALTER TABLE users ADD COLUMN admin_permissions JSON NULL",
]


def _get_mysql_params() -> Optional[dict]:
    """
    从环境变量读取 MySQL 连接配置。
    需要 DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME。
    任意缺失则返回 None。
    """
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    name = os.getenv("DB_NAME")
    port = int(os.getenv("DB_PORT", "3306") or 3306)
    if not all([host, user, password, name]):
        return None
    return {
        "host": host,
        "user": user,
        "password": password,
        "database": name,
        "port": port,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor if pymysql else None,
    }


@contextmanager
def mysql_conn():
    """
    提供一个简单的 MySQL 连接上下文。

    用法：
        with mysql_conn() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    if pymysql is None:
        logger.warning("pymysql 未安装，跳过数据库操作。请运行: pip install pymysql")
        yield None
        return
    params = _get_mysql_params()
    if params is None:
        logger.info("未配置 DB_* 环境变量，跳过数据库连接。")
        yield None
        return
    conn = None
    try:
        conn = pymysql.connect(**params)
        yield conn
    except Exception as e:  # pragma: no cover
        logger.error(f"连接 MySQL 失败: {e}")
        yield None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _bert_master_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_default_risk_lexicon_rows() -> List[Tuple[str, str]]:
    """直接加载 chinese_emotion/sensitive_words.py，避免 import chinese_emotion 包拉取 transformers。"""
    path = os.path.join(_bert_master_root(), "chinese_emotion", "sensitive_words.py")
    if not os.path.isfile(path):
        return []
    spec = importlib.util.spec_from_file_location("_cm_sensitive_words_seed", path)
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "iter_default_lexicon_rows", None)
    if not callable(fn):
        return []
    return list(fn())


def _seed_risk_lexicon_if_empty(cur) -> None:
    """首次部署：将内置词表与默认阈值写入数据库，便于后续在库内维护。"""
    try:
        cur.execute("SELECT COUNT(*) AS c FROM risk_lexicon")
        row = cur.fetchone() or {}
        if int(row.get("c") or 0) > 0:
            return
    except Exception:
        return
    rows = _load_default_risk_lexicon_rows()
    for phrase, category in rows:
        try:
            cur.execute(
                "INSERT INTO risk_lexicon (phrase, category, is_active, sort_order) VALUES (%s,%s,TRUE,0)",
                (phrase, category),
            )
        except Exception as e:
            logger.warning(f"写入 risk_lexicon 种子失败: {e}")
    defaults = [
        ("attention_threshold", "0.35"),
        ("intervention_threshold", "0.52"),
    ]
    for k, v in defaults:
        try:
            cur.execute(
                "INSERT INTO risk_config (config_key, config_value) VALUES (%s,%s)",
                (k, v),
            )
        except Exception as e:
            logger.warning(f"写入 risk_config 种子失败: {e}")
    if rows:
        logger.info("已写入风险词表与阈值种子数据（可在表 risk_lexicon / risk_config 中维护）")


def init_tables():
    """启动时自动建表（幂等），若未配置数据库则静默跳过。"""
    global _TABLES_INITED
    if _TABLES_INITED:
        return
    with mysql_conn() as conn:
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                for sql in _CREATE_TABLES_SQL:
                    cur.execute(sql)
                for sql in _ADD_COLUMNS_SQL:
                    try:
                        cur.execute(sql)
                    except Exception:
                        pass  # column already exists / already varchar
                try:
                    cur.execute(
                        "UPDATE users SET role=%s WHERE role=%s",
                        ("school_admin", "admin"),
                    )
                except Exception:
                    pass
                _seed_risk_lexicon_if_empty(cur)
            conn.commit()
            _TABLES_INITED = True
            logger.info("数据库表初始化完成")
        except Exception as e:
            logger.warning(f"建表失败: {e}")
