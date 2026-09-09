import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path


class AuthService:
    COOKIE_NAME = "tc_session"
    SESSION_SECONDS = 8 * 60 * 60

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "users.db"
        self.secret_path = self.data_dir / ".session_secret"
        self.secret = self._load_secret()
        self._initialize()

    def _load_secret(self) -> bytes:
        if self.secret_path.exists():
            return self.secret_path.read_bytes()
        secret = secrets.token_bytes(32)
        self.secret_path.write_bytes(secret)
        return secret

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','admin')),
                display_name TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
                )"""
            )

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
        return f"scrypt${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"

    @classmethod
    def _verify_password(cls, password: str, stored: str) -> bool:
        try:
            _, salt_text, digest_text = stored.split("$", 2)
            salt = base64.urlsafe_b64decode(salt_text)
            expected = base64.urlsafe_b64decode(digest_text)
            actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def needs_setup(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0

    def create_user(self, username: str, password: str, display_name: str, role: str = "user"):
        username = username.strip().lower()
        display_name = display_name.strip()
        if not 3 <= len(username) <= 40 or not username.replace("_", "").replace("-", "").isalnum():
            raise ValueError("帳號需為3至40個英數字，可使用底線或連字號")
        if len(password) < 8:
            raise ValueError("密碼至少需要8個字元")
        if role not in {"user", "admin"}:
            raise ValueError("無效的角色")
        if not display_name:
            raise ValueError("請輸入顯示名稱")
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users(username,password_hash,role,display_name,created_at) VALUES(?,?,?,?,?)",
                    (username, self._hash_password(password), role, display_name, int(time.time())),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("帳號已存在") from exc
        return {"username": username, "display_name": display_name, "role": role}

    def authenticate(self, username: str, password: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username,password_hash,role,display_name,active FROM users WHERE username=?",
                (username.strip().lower(),),
            ).fetchone()
        if not row or not row["active"] or not self._verify_password(password, row["password_hash"]):
            return None
        return {"username": row["username"], "display_name": row["display_name"], "role": row["role"]}

    def issue_token(self, user: dict) -> str:
        payload = {"u": user["username"], "n": user["display_name"], "r": user["role"], "exp": int(time.time()) + self.SESSION_SECONDS}
        raw = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).rstrip(b"=")
        signature = base64.urlsafe_b64encode(hmac.new(self.secret, raw, hashlib.sha256).digest()).rstrip(b"=")
        return f"{raw.decode()}.{signature.decode()}"

    def read_token(self, token: str | None):
        if not token:
            return None
        try:
            raw_text, signature_text = token.split(".", 1)
            raw = raw_text.encode()
            expected = base64.urlsafe_b64encode(hmac.new(self.secret, raw, hashlib.sha256).digest()).rstrip(b"=").decode()
            if not hmac.compare_digest(signature_text, expected):
                return None
            payload = json.loads(base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4)))
            if int(payload["exp"]) < int(time.time()):
                return None
            return {"username": payload["u"], "display_name": payload["n"], "role": payload["r"]}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def list_users(self):
        with self._connect() as connection:
            rows = connection.execute("SELECT username,display_name,role,active,created_at FROM users ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]
