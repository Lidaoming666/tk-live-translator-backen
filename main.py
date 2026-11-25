import asyncio
import os
import time
from collections import deque
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from TikTokLive import TikTokLiveClient
from TikTokLive.events import CommentEvent, ConnectEvent, DisconnectEvent

app = FastAPI(title="TK Live Translator Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Pages URL before company-wide use.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

events = deque(maxlen=500)
client: Optional[TikTokLiveClient] = None
client_task: Optional[asyncio.Task] = None
active_username: Optional[str] = None
connected = False
last_error: Optional[str] = None
lock = asyncio.Lock()

class ConnectRequest(BaseModel):
    unique_id: str
    comments_only: bool = True

async def translate_to_chinese(text: str) -> str:
    # MVP translation path; subject to MyMemory availability and rate limits.
    try:
        async with httpx.AsyncClient(timeout=8) as session:
            response = await session.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text[:450], "langpair": "en|zh-CN"},
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("responseData") or {}).get("translatedText") or ""
    except Exception:
        return ""

async def on_comment(event: CommentEvent):
    raw = (getattr(event, "comment", "") or "").strip()
    if not raw:
        return
    user = getattr(getattr(event, "user", None), "nickname", None) or "观众"
    translation = await translate_to_chinese(raw)
    events.appendleft({
        "user": str(user),
        "text": raw,
        "translation": translation,
        "language": "原文",
        "time": time.strftime("%H:%M:%S"),
    })

@app.get("/api/health")
async def health():
    return {"ok": True, "connected": connected, "username": active_username, "error": last_error}

@app.post("/api/live/connect")
async def connect_live(body: ConnectRequest):
    global client, client_task, active_username, connected, last_error
    username = body.unique_id.strip().lstrip("@")
    if not username:
        raise HTTPException(400, "请输入 TikTok 主播账号")
    async with lock:
        if client_task and not client_task.done():
            try:
                if client:
                    await client.stop()
            except Exception:
                pass
            client_task.cancel()
        events.clear()
        connected = False
        last_error = None
        active_username = username
        client = TikTokLiveClient(unique_id=username)
        client.add_listener(CommentEvent, on_comment)

        async def runner():
            global connected, last_error
            try:
                await client.start()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                connected = False
            finally:
                connected = False

        client_task = asyncio.create_task(runner())
    # Give the connector a moment to resolve the public room and handshake.
    await asyncio.sleep(1.5)
    if client and getattr(client, "connected", False):
        connected = True
        return {"connected": True, "message": f"已连接 @{username}，正在等待新评论"}
    if last_error:
        return {"connected": False, "message": last_error}
    return {"connected": False, "message": "正在连接；如果主播未开播或签名服务限流，请稍后重试"}

@app.get("/api/live/comments")
async def get_comments():
    return {"comments": list(events), "connected": connected, "username": active_username}

@app.post("/api/live/disconnect")
async def disconnect_live():
    global client, client_task, connected, active_username
    async with lock:
        if client:
            try:
                await client.stop()
            except Exception:
                pass
        if client_task and not client_task.done():
            client_task.cancel()
        client = None
        client_task = None
        connected = False
        active_username = None
    return {"ok": True}
