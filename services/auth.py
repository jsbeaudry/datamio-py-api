"""
API Key Authentication Service
Handles generation, validation, and management of API keys
"""
import os
import secrets
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader

# Admin API key from environment (survives redeployments)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# Storage file for API keys
API_KEYS_FILE = Path(__file__).parent.parent / "api_keys.json"

# API Key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _hash_key(key: str) -> str:
    """Hash an API key for secure storage"""
    return hashlib.sha256(key.encode()).hexdigest()


def _load_keys() -> Dict:
    """Load API keys from storage"""
    if API_KEYS_FILE.exists():
        with open(API_KEYS_FILE, "r") as f:
            return json.load(f)
    return {"keys": {}}


def _save_keys(data: Dict) -> None:
    """Save API keys to storage"""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def generate_api_key(name: str, description: str = "") -> Dict:
    """
    Generate a new API key for a user/service.
    Returns the full key (only shown once) and metadata.
    """
    # Generate a secure random key with prefix for identification
    raw_key = f"datamio_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    key_id = secrets.token_hex(8)

    key_data = {
        "id": key_id,
        "name": name,
        "description": description,
        "key_prefix": raw_key[:12] + "...",  # For identification without exposing full key
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None,
        "is_active": True,
    }

    # Load existing keys and add new one
    data = _load_keys()
    data["keys"][key_hash] = key_data
    _save_keys(data)

    return {
        "api_key": raw_key,  # Only returned once at creation time
        "key_id": key_id,
        "name": name,
        "key_prefix": key_data["key_prefix"],
        "message": "Store this API key securely. It will not be shown again."
    }


def validate_api_key(api_key: str) -> Optional[Dict]:
    """
    Validate an API key and return its metadata if valid.
    Updates last_used timestamp on successful validation.
    """
    if not api_key:
        return None

    # Check admin key from environment first (always valid, survives redeployments)
    if ADMIN_API_KEY and api_key == ADMIN_API_KEY:
        return {
            "id": "admin",
            "name": "admin",
            "description": "Environment admin key",
            "is_active": True,
            "is_admin": True,
        }

    key_hash = _hash_key(api_key)
    data = _load_keys()

    if key_hash in data["keys"]:
        key_data = data["keys"][key_hash]
        if key_data.get("is_active", True):
            # Update last used timestamp
            key_data["last_used"] = datetime.utcnow().isoformat()
            _save_keys(data)
            return key_data

    return None


def revoke_api_key(key_id: str) -> bool:
    """Revoke an API key by its ID"""
    data = _load_keys()

    for key_hash, key_data in data["keys"].items():
        if key_data["id"] == key_id:
            key_data["is_active"] = False
            key_data["revoked_at"] = datetime.utcnow().isoformat()
            _save_keys(data)
            return True

    return False


def delete_api_key(key_id: str) -> bool:
    """Permanently delete an API key by its ID"""
    data = _load_keys()

    for key_hash, key_data in list(data["keys"].items()):
        if key_data["id"] == key_id:
            del data["keys"][key_hash]
            _save_keys(data)
            return True

    return False


def list_api_keys() -> List[Dict]:
    """List all API keys (without the actual key values)"""
    data = _load_keys()
    return list(data["keys"].values())


def get_api_key_by_id(key_id: str) -> Optional[Dict]:
    """Get API key metadata by ID"""
    data = _load_keys()

    for key_data in data["keys"].values():
        if key_data["id"] == key_id:
            return key_data

    return None


# FastAPI dependency for API key validation
async def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> Dict:
    """
    FastAPI dependency that requires a valid API key.
    Use this as a dependency on protected routes.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'X-API-Key' header.",
        )

    key_data = validate_api_key(api_key)

    if not key_data:
        raise HTTPException(
            status_code=403,
            detail="Invalid or revoked API key.",
        )

    return key_data
