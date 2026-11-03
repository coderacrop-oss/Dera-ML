from __future__ import annotations

import base64
import json

from .errors import ApiError


def encode_cursor(eid: str, etag: str, after: str) -> str:
    raw = json.dumps({"e": eid, "t": etag, "a": after}, ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict:
    try:
        doc = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("utf-8"))
        if not (isinstance(doc, dict) and all(isinstance(doc.get(k), str) for k in ("e", "t", "a"))):
            raise ValueError("shape")
        return doc
    except Exception as exc:  # any malformed cursor
        raise ApiError(422, "invalid_cursor", "cursor ไม่ถูกต้อง — ใช้ค่า next_cursor ที่ได้จากหน้าก่อนหน้าเท่านั้น") from exc
