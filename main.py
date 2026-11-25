import asyncio
import logging
import os
import time
from collections import deque
from contextlib import suppress
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langdetect import detect, LangDetectException
from pydantic import BaseModel, Field
from TikTokLive import TikTokLiveClient
from TikTokLive.events import CommentEvent, ConnectEvent, DisconnectEvent

app = FastAPI(title="TK Live Translator Bridge", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend origin before public use.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

events = deque(maxlen=500)
client: Optional[TikTokLiveClient] = None
client_task: Optional[asyncio.Task] = None
active_username: Optional[str] = None
connected = False
connection_status = "idle"
last_error: Optional[str] = None
last_comment_at: Optional[str] = None
lock = asyncio.Lock()
translation_semaphore = asyncio.Semaphore(3)
translation_cache = {}


class ConnectRequest(BaseModel):
    unique_id: str = Field(min_length=1, max_length=200)
    comments_only: bool = True


class TranslateTestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    source_language: str = "auto"


def detect_source_language(text: str) -> str:
    """MyMemory requires an explicit source language; detect it locally."""
    try:
        detected = detect(text)
    except LangDetectException:
        detected = "en"
    except Exception:
        detected = "en"

    normalized = (detected or "en").lower()
    aliases = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "pt-br": "pt-BR",
    }
    return aliases.get(normalized, normalized)


def trim_to_mymemory_limit(text: str, max_bytes: int = 450) -> str:
    """Keep requests below MyMemory's documented 500-byte query limit."""
    result = text
    while result and len(result.encode("utf-8")) > max_bytes:
        result = result[:-1]
    return result


async def translate_to_chinese(
    text: str, requested_source: str = "auto"
) -> tuple[str, str, str, Optional[str]]:
    raw = (text or "").strip()
    if not raw:
        return "", "und", "skipped", "Empty comment"

    if requested_source and requested_source.lower() != "auto":
        source_language = requested_source
    elif len(raw) < 3:
        source_language = "en"
    else:
        source_language = detect_source_language(raw)

    # Chinese comments already match the target language; keep them visible.
    if source_language.lower().startswith("zh"):
        return raw, source_language, "not_needed", None

    cache_key = (source_language.lower(), raw)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    query_text = trim_to_mymemory_limit(raw)
    if not query_text:
        return "", source_language, "failed", "Comment is too large to translate"

    params = {
        "q": query_text,
        "langpair": f"{source_language}|zh-CN",
        "mt": "1",
    }
    contact_email = os.getenv("MYMEMORY_EMAIL", "").strip()
    if contact_email:
        params["de"] = contact_email

    async with translation_semaphore:
        try:
            async with httpx.AsyncClient(timeout=12) as session:
                response = await session.get(
                    "https://api.mymemory.translated.net/get",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

            provider_status = int(data.get("responseStatus", 0))
            if provider_status != 200:
                details = data.get("responseDetails") or "Translation provider rejected the request"
                raise RuntimeError(f"MyMemory {provider_status}: {details}")

            translated = ((data.get("responseData") or {}).get("translatedText") or "").strip()
            if not translated:
                raise RuntimeError("MyMemory returned an empty translation")

            result = (translated, source_language, "translated", None)
            if len(translation_cache) > 1000:
                translation_cache.clear()
            translation_cache[cache_key] = result
            return result
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"[:240]
            return "", source_language, "failed", error_message


@app.get("/")
async def root():
    return {
        "service": "TK Live Translator Bridge",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health():
    actual_connected = bool(client and getattr(client, "connected", False))
    return {
        "ok": True,
        "connected": actual_connected or connected,
        "connection_status": connection_status,
        "username": active_username,
        "last_comment_at": last_comment_at,
        "comments_buffered": len(events),
        "error": last_error,
    }


@app.get("/api/live/status")
async def live_status():
    actual_connected = bool(client and getattr(client, "connected", False))
    return {
        "connected": actual_connected or connected,
        "status": connection_status,
        "username": active_username,
        "last_comment_at": last_comment_at,
        "comments_buffered": len(events),
        "error": last_error,
    }


@app.post("/api/live/connect")
async def connect_live(body: ConnectRequest):
    global client, client_task, active_username, connected
    global connection_status, last_error, last_comment_at

    username = body.unique_id.strip().lstrip("@")
    if not username:
        raise HTTPException(400, "请输入 TikTok 主播用户名")

    async with lock:
        old_client = client
        old_task = client_task
        client = None
        client_task = None

        if old_client is not None:
            try:
                await asyncio.wait_for(old_client.disconnect(), timeout=5)
            except Exception:
                pass
        if old_task is not None and not old_task.done():
            old_task.cancel()
            with suppress(asyncio.CancelledError):
                await old_task

        events.clear()
        connected = False
        connection_status = "connecting"
        last_error = None
        last_comment_at = None
        active_username = username

        current_client = TikTokLiveClient(unique_id=username)
        current_client.logger.setLevel(logging.INFO)
        client = current_client

        async def on_connect(event: ConnectEvent):
            global connected, connection_status, last_error
            if client is current_client:
                connected = True
                connection_status = "connected"
                last_error = None

        async def on_disconnect(event: DisconnectEvent):
            global connected, connection_status
            if client is current_client:
                connected = False
                if connection_status != "error":
                    connection_status = "disconnected"

        async def on_comment(event: CommentEvent):
            global connected, connection_status, last_comment_at
            if client is not current_client:
                return

            raw = (getattr(event, "comment", "") or "").strip()
            if not raw:
                return

            # Receiving a comment is direct evidence that the websocket is active.
            connected = True
            connection_status = "connected"

            user_obj = getattr(event, "user", None)
            nickname = getattr(user_obj, "nickname", None) or "观众"
            translation, source_language, translation_status, translation_error = (
                await translate_to_chinese(raw)
            )
            received_at = time.strftime("%Y-%m-%d %H:%M:%S")
            last_comment_at = received_at
            item = {
                "user": str(nickname),
                "text": raw,
                "translation": translation,
                "language": "原文",
                "source_language": source_language,
                "translation_status": translation_status,
                "translation_error": translation_error,
                "time": received_at,
            }
            events.appendleft(item)

        current_client.add_listener(ConnectEvent, on_connect)
        current_client.add_listener(DisconnectEvent, on_disconnect)
        current_client.add_listener(CommentEvent, on_comment)

        async def run_client():
            global connected, connection_status, last_error
            try:
                # connect() waits for the live session to end. Keeping this task
                # alive avoids the previous bug where start() returned its inner
                # task immediately and the wrapper marked the service disconnected.
                await current_client.connect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if client is current_client:
                    connected = False
                    connection_status = "error"
                    last_error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                if client is current_client and connection_status not in ("error", "idle"):
                    connected = False
                    connection_status = "disconnected"

        client_task = asyncio.create_task(run_client())

    # Allow the client to resolve the room and establish its websocket.
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        actual_connected = bool(getattr(current_client, "connected", False))
        if actual_connected:
            connected = True
            connection_status = "connected"
        if connection_status in ("connected", "error", "disconnected"):
            break
        await asyncio.sleep(0.25)

    if connection_status == "connected":
        return {
            "connected": True,
            "status": "connected",
            "message": f"已连接 @{username}，正在接收直播评论",
        }
    if connection_status == "error":
        return {
            "connected": False,
            "status": "error",
            "message": last_error or "TikTok 直播连接失败",
        }
    return {
        "connected": False,
        "status": "connecting",
        "message": "连接请求已提交，仍在等待 TikTok WebSocket 建立。请稍后查看 /api/live/status。",
    }


@app.get("/api/live/comments")
async def get_comments():
    actual_connected = bool(client and getattr(client, "connected", False))
    return {
        "comments": list(events),
        "connected": actual_connected or connected,
        "status": connection_status,
        "username": active_username,
        "last_comment_at": last_comment_at,
    }


@app.post("/api/live/disconnect")
async def disconnect_live():
    global client, client_task, connected, active_username
    global connection_status, last_error, last_comment_at

    async with lock:
        old_client = client
        old_task = client_task
        client = None
        client_task = None
        connected = False
        active_username = None
        connection_status = "idle"
        last_comment_at = None

        if old_client is not None:
            try:
                await asyncio.wait_for(old_client.disconnect(), timeout=5)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:240]
        if old_task is not None and not old_task.done():
            old_task.cancel()
            with suppress(asyncio.CancelledError):
                await old_task

    return {"ok": True, "status": "disconnected"}


@app.post("/api/translate/test")
async def translate_test(body: TranslateTestRequest):
    translation, source_language, status, error = await translate_to_chinese(
        body.text, body.source_language
    )
    return {
        "original": body.text,
        "source_language": source_language,
        "target_language": "zh-CN",
        "translation": translation,
        "status": status,
        "error": error,
    }
