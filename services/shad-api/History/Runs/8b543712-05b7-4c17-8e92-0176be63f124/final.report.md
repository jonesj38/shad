# Run Report: 8b543712-05b7-4c17-8e92-0176be63f124

**Goal:** Build a simple REST API with user registration, login, and profile endpoints using Flask

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 1884

## Result

Here's a complete Flask REST API with user registration, login, and profile endpoints:

```python
"""Simple Flask REST API with user registration, login, and profile endpoints."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable

from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(32)

# In-memory storage (use a real database in production)
users_db: dict[str, dict[str, Any]] = {}
tokens_db: dict[str, dict[str, Any]] = {}

TOKEN_EXPIRY_HOURS = 24


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    """Verify password against stored hash."""
    salt, hashed = stored.split(":")
    return hashlib.sha256((password + salt).encode()).hexdigest() == hashed


def generate_token(user_id: str) -> str:
    """Generate authentication token."""
    token = secrets.token_urlsafe(32)
    tokens_db[token] = {
        "user_id": user_id,
        "expires": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return token


def require_auth(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        token_data = tokens_db.get(token)

        if not token_data:
            return jsonify({"error": "Invalid token"}), 401

        if datetime.now(timezone.utc) > token_data["expires"]:
            del tokens_db[token]
            return jsonify({"error": "Token expired"}), 401

        request.user_id = token_data["user_id"]
        return f(*args, **kwargs)
    return decorated


@app.route("/api/register", methods=["POST"])
def register() -> tuple[Any, int]:
    """Register a new user."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required"}), 400

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if "@" not in email or "." not in email:
        return jsonify({"error": "Invalid email format"}), 400

    if username in users_db:
        return jsonify({"error": "Username already taken"}), 409

    if any(u["email"] == email for u in users_db.values()):
        return jsonify({"error": "Email already registered"}), 409

    users_db[username] = {
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bio": "",
    }

    return jsonify({
        "message": "User registered successfully",
        "user": {"username": username, "email": email},
    }), 201


@app.route("/api/login", methods=["POST"])
def login() -> tuple[Any, int]:
    """Login and get authentication token."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = users_db.get(username)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid username or password"}), 401

    token = generate_token(username)

    return jsonify({
        "message": "Login successful",
        "token": token,
        "expires_in": TOKEN_EXPIRY_HOURS * 3600,
    }), 200


@app.route("/api/logout", methods=["POST"])
@require_auth
def logout() -> tuple[Any, int]:
    """Logout and invalidate token."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.split(" ", 1)[1]
    if token in tokens_db:
        del tokens_db[token]
    return jsonify({"message": "Logged out successfully"}), 200


@app.route("/api/profile", methods=["GET"])
@require_auth
def get_profile() -> tuple[Any, int]:
    """Get current user's profile."""
    user = users_db.get(request.user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "username": user["username"],
        "email": user["email"],
        "bio": user["bio"],
        "created_at": user["created_at"],
    }), 200


@app.route("/api/profile", methods=["PUT"])
@require_auth
def update_profile() -> tuple[Any, int]:
    """Update current user's profile."""
    user = users_db.get(request.user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    if "email" in data:
        email = data["email"].strip().lower()
        if "@" not in email or "." not in email:
            return jsonify({"error": "Invalid email format"}), 400
        if email != user["email"] and any(u["email"] == email for u in users_db.values()):
            return jsonify({"error": "Email already registered"}), 409
        user["email"] = email

    if "bio" in data:
        user["bio"] = data["bio"][:500]

    return jsonify({
        "message": "Profile updated successfully",
        "profile": {"username": user["username"], "email": user["email"], "bio": user["bio"]},
    }), 200


@app.route("/api/profile/password", methods=["PUT"])
@require_auth
def change_password() -> tuple[Any, int]:
    """Change current user's password."""
    user = users_db.get(request.user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if not verify_password(current_password, user["password_hash"]):
        return jsonify({"error": "Current password is incorrect"}), 401

    if len(new_password) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400

    user["password_hash"] = hash_password(new_password)

    # Invalidate all tokens for this user
    for token in [t for t, d in tokens_db.items() if d["user_id"] == request.user_id]:
        del tokens_db[token]

    return jsonify({"message": "Password changed successfully. Please login again."}), 200


@app.route("/api/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "flask-auth-api"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/register` | POST | No | Register new user |
| `/api/login` | POST | No | Login, get token |
| `/api/logout` | POST | Yes | Invalidate token |
| `/api/profile` | GET | Yes | Get user profile |
| `/api/profile` | PUT | Yes | Update profile |
| `/api/profile/password` | PUT | Yes | Change password |
| `/api/health` | GET | No | Health check |

## Usage Examples

```bash
# Install Flask
pip install flask

# Run the server
python flask_auth.py

# Register
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "email": "john@example.com", "password": "secret123"}'

# Login
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "password": "secret123"}'

# Get profile (use token from login response)
curl http://localhost:5000/api/profile \
  -H "Authorization: Bearer <token>"

# Update profile
curl -X PUT http://localhost:5000/api/profile \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"bio": "Hello world"}'
```

**Note**: This uses in-memory storage and SHA-256 for simplicity. For production, use a real database and `bcrypt` or `argon2` for password hashing.
