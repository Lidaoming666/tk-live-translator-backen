import asyncio
import logging
import os
import re
import time
from collections import deque
from contextlib import suppress
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from TikTokLive import TikTokLiveClient
from TikTokLive.events import CommentEvent, ConnectEvent, DisconnectEvent

try:
    from langdetect import detect, LangDetectException
except Exception:  # The script still works if langdetect is unavailable.
    detect = None
    LangDetectException = Exception

APP_VERSION = "0.3.1"
MYMEMORY_URL = "https://api.mymemory.translated.net/get"
AZURE_TRANSLATOR_URL = os.getenv(
    "AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com"
).rstrip("/")
TRANSLATE_INTERVAL_SECONDS = max(
    0.5, float(os.getenv("TRANSLATE_INTERVAL_SECONDS", "1.0"))
)
TRANSLATE_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("TRANSLATE_TIMEOUT_SECONDS", "12"))
)
MAX_PENDING_TRANSLATIONS = max(
    20, int(os.getenv("MAX_PENDING_TRANSLATIONS", "120"))
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tk-live-translator")

app = FastAPI(title="TK Live Translator Bridge", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your frontend origin before public production use.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Latest comments appear first. Each dict is updated in place as translation finishes.
events = deque(maxlen=500)
client: Optional[TikTokLiveClient] = None
client_task: Optional[asyncio.Task] = None
translation_worker_task: Optional[asyncio.Task] = None
active_username: Optional[str] = None
connected = False
connection_status = "idle"
last_error: Optional[str] = None
last_comment_at: Optional[str] = None
connect_lock = asyncio.Lock()
translation_queue: Optional[asyncio.Queue] = None
translation_cache: dict[tuple[str, str], tuple[str, str, str, Optional[str]]] = {}
http_client: Optional[httpx.AsyncClient] = None


class ConnectRequest(BaseModel):
    unique_id: str = Field(min_length=1, max_length=200)
    comments_only: bool = True


class TranslateTestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    source_language: str = "auto"


def detect_script_language(text: str) -> Optional[str]:
    """Detect CJK scripts before statistical detection to avoid zh -> ko mistakes."""
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af\u1100-\u11ff]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text):
        traditional_chars = set(
            "體國門學車書風龍後發臺灣萬與這個為來時會說對裡還讓點經過實現開關聽見歡樂網絡電腦號碼"
        )
        return "zh-TW" if any(ch in traditional_chars for ch in text) else "zh-CN"
    return None


def normalize_language_code(code: Optional[str]) -> str:
    value = (code or "und").strip().lower().replace("_", "-")
    aliases = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-hans": "zh-CN",
        "zh-tw": "zh-TW",
        "zh-hant": "zh-TW",
        "iw": "he",
        "in": "id",
        "pt-br": "pt-BR",
    }
    return aliases.get(value, value)


def detect_source_language(text: str) -> str:
    script = detect_script_language(text)
    if script:
        return script
    if detect is not None and len(text.strip()) >= 5:
        try:
            return normalize_language_code(detect(text))
        except (LangDetectException, Exception):
            pass
    # Short Latin-script comments are commonly English; Google/Microsoft/MyMemory
    # can still handle many short phrases, but this is only a fallback hint.
    return "en"


def trim_to_mymemory_limit(text: str, max_bytes: int = 450) -> str:
    """MyMemory documents a 500-byte maximum for the q parameter."""
    result = text
    while result and len(result.encode("utf-8")) > max_bytes:
        result = result[:-1]
    return result


async def translate_with_azure(text: str, source_language: str) -> tuple[str, str]:
    key = os.getenv("AZURE_TRANSLATOR_KEY", "").strip()
    region = os.getenv("AZURE_TRANSLATOR_REGION", "").strip()
    if not key or not region:
        raise RuntimeError("Azure Translator 未配置 Key/Region")
    if http_client is None:
        raise RuntimeError("HTTP 客户端尚未启动")

    params = {"api-version": "3.0", "to": "zh-Hans"}
    if source_language not in ("auto", "und", ""):
        params["from"] = source_language
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-Type": "application/json; charset=UTF-8",
    }
    response = await http_client.post(
        f"{AZURE_TRANSLATOR_URL}/translate",
        params=params,
        headers=headers,
        json=[{"Text": text}],
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("Azure Translator 返回了空数据")
    item = data[0]
    translations = item.get("translations") or []
    translated = (translations[0].get("text") or "").strip() if translations else ""
    detected = normalize_language_code(
        (item.get("detectedLanguage") or {}).get("language") or source_language
    )
    if not translated:
        raise RuntimeError("Azure Translator 返回了空译文")
    return translated, detected


async def translate_with_mymemory(text: str, source_language: str) -> tuple[str, str]:
    if http_client is None:
        raise RuntimeError("HTTP 客户端尚未启动")
    query_text = trim_to_mymemory_limit(text)
    if not query_text:
        raise RuntimeError("评论内容超过可翻译长度")

    params = {
        "q": query_text,
        "langpair": f"{source_language}|zh-CN",
        "mt": "1",
    }
    # Set MYMEMORY_EMAIL in Render to raise the documented free daily quota.
    # Never put an email address or API secret directly in source code.
    contact_email = os.getenv("MYMEMORY_EMAIL", "").strip()
    if contact_email:
        params["de"] = contact_email

    response = await http_client.get(MYMEMORY_URL, params=params)
    response.raise_for_status()
    data = response.json()
    provider_status = int(data.get("responseStatus", 0))
    if provider_status != 200:
        details = data.get("responseDetails") or "MyMemory 翻译服务拒绝了请求"
        raise RuntimeError(f"MyMemory {provider_status}: {details}")
    translated = ((data.get("responseData") or {}).get("translatedText") or "").strip()
    if not translated:
        raise RuntimeError("MyMemory 返回了空译文")
    return translated, source_language


async def translate_to_chinese(
    text: str, requested_source: str = "auto"
) -> tuple[str, str, str, Optional[str]]:
    raw = (text or "").strip()
    if not raw:
        return "", "und", "skipped", "评论内容为空"

    source_language = (
        normalize_language_code(requested_source)
        if requested_source and requested_source.lower() != "auto"
        else detect_source_language(raw)
    )

    # Chinese comments are already in the target language; do not send them out.
    if source_language.lower().startswith("zh"):
        return raw, source_language, "unchanged", None

    cache_key = (source_language, raw)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        if os.getenv("AZURE_TRANSLATOR_KEY", "").strip():
            translated, detected = await translate_with_azure(raw, source_language)
            result = (translated, detected, "translated", None)
        else:
            translated, detected = await translate_with_mymemory(raw, source_language)
            result = (translated, detected, "translated", None)

        if len(translation_cache) >= 2000:
            translation_cache.clear()
        translation_cache[cache_key] = result
        return result
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code == 429:
            if os.getenv("AZURE_TRANSLATOR_KEY", "").strip():
                detail = "Azure Translator 返回 429：请检查 F0 配额或请求频率"
            elif os.getenv("MYMEMORY_EMAIL", "").strip():
                detail = "MyMemory 返回 429：已配置邮箱仍触发限流，请稍后再试"
            else:
                detail = "MyMemory 返回 429：匿名额度约 5,000 字符/天；在 Render 设置 MYMEMORY_EMAIL 可提高到约 50,000 字符/天"
        else:
            detail = f"翻译服务 HTTP {status_code}"
        logger.warning("Translation provider HTTP %s", status_code)
        return "", source_language, "failed", detail
    except httpx.TimeoutException:
        return "", source_language, "failed", "翻译服务请求超时"
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return "", source_language, "failed", f"{type(exc).__name__}: {str(exc)[:180]}"


@app.on_event("startup")
async def startup() -> None:
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(TRANSLATE_TIMEOUT_SECONDS),
        headers={"Accept": "application/json", "User-Agent": "TKLiveTranslator/0.3.1"},
        follow_redirects=True,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None


async def translation_worker() -> None:
    """Process translations one at a time to avoid burst requests."""
    global translation_queue
    last_started = 0.0
    while True:
        if translation_queue is None:
            await asyncio.sleep(0.2)
            continue
        item = await translation_queue.get()
        try:
            wait = TRANSLATE_INTERVAL_SECONDS - (time.monotonic() - last_started)
            if wait > 0:
                await asyncio.sleep(wait)
            last_started = time.monotonic()
            item["translation_status"] = "translating"
            translation, source_language, status, error = await translate_to_chinese(
                item["text"]
            )
            item["translation"] = translation
            item["source_language"] = source_language
            item["language"] = source_language
            item["translation_status"] = status
            item["translation_error"] = error
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Unexpected translation worker error")
            item["translation_status"] = "failed"
            item["translation_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        finally:
            translation_queue.task_done()


@app.get("/")
async def root():
    return {
        "service": "TK Live Translator Bridge",
        "version": APP_VERSION,
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
        "translation_queue": translation_queue.qsize() if translation_queue else 0,
        "translation_provider": "azure" if os.getenv("AZURE_TRANSLATOR_KEY") else "mymemory",
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
        "translation_queue": translation_queue.qsize() if translation_queue else 0,
        "error": last_error,
    }


@app.post("/api/live/connect")
async def connect_live(body: ConnectRequest):
    global client, client_task, translation_worker_task, translation_queue
    global active_username, connected, connection_status, last_error, last_comment_at

    username = body.unique_id.strip().lstrip("@")
    if not username:
        raise HTTPException(status_code=400, detail="请输入 TikTok 主播用户名")

    async with connect_lock:
        old_client = client
        old_task = client_task
        old_worker = translation_worker_task
        client = None
        client_task = None
        translation_worker_task = None

        if old_client is not None:
            try:
                await asyncio.wait_for(old_client.disconnect(), timeout=5)
            except Exception:
                logger.info("Previous TikTok client was already disconnected")
        for task in (old_task, old_worker):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

        events.clear()
        translation_cache.clear()
        translation_queue = asyncio.Queue(maxsize=MAX_PENDING_TRANSLATIONS)
        connected = False
        connection_status = "connecting"
        last_error = None
        last_comment_at = None
        active_username = username

        current_client = TikTokLiveClient(unique_id=username)
        current_client.logger.setLevel(logging.INFO)
        client = current_client
        translation_worker_task = asyncio.create_task(translation_worker())

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

            connected = True
            connection_status = "connected"
            user_obj = getattr(event, "user", None)
            nickname = getattr(user_obj, "nickname", None) or "观众"
            received_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            last_comment_at = received_at
            item = {
                "user": str(nickname),
                "text": raw,
                "translation": "",
                "language": "auto",
                "source_language": "auto",
                "translation_status": "queued",
                "translation_error": None,
                "time": received_at,
            }
            events.appendleft(item)

            # Chinese comments can be marked immediately without using a provider.
            detected = detect_source_language(raw)
            if detected.lower().startswith("zh"):
                item["translation"] = raw
                item["language"] = detected
                item["source_language"] = detected
                item["translation_status"] = "unchanged"
                return

            if translation_queue is not None:
                try:
                    translation_queue.put_nowait(item)
                except asyncio.QueueFull:
                    item["translation_status"] = "skipped_queue_full"
                    item["translation_error"] = "评论过多，翻译队列已满；原文仍已保存"

        current_client.add_listener(ConnectEvent, on_connect)
        current_client.add_listener(DisconnectEvent, on_disconnect)
        current_client.add_listener(CommentEvent, on_comment)

        async def run_client():
            global connected, connection_status, last_error
            try:
                await current_client.connect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if client is current_client:
                    connected = False
                    connection_status = "error"
                    last_error = f"{type(exc).__name__}: {str(exc)}"[:500]
                    logger.exception("TikTok LIVE connection failed")
            finally:
                if client is current_client and connection_status not in ("error", "idle"):
                    connected = False
                    connection_status = "disconnected"

        client_task = asyncio.create_task(run_client())

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if bool(getattr(current_client, "connected", False)):
            connected = True
            connection_status = "connected"
        if connection_status in ("connected", "error", "disconnected"):
            break
        await asyncio.sleep(0.25)

    if connection_status == "connected":
        return {
            "connected": True,
            "status": "connected",
            "message": f"已连接 @{username}，正在接收直播评论并排队翻译",
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
        "message": "连接请求已提交，仍在等待 TikTok WebSocket 建立。请查看 /api/live/status。",
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
        "translation_queue": translation_queue.qsize() if translation_queue else 0,
    }


@app.post("/api/live/disconnect")
async def disconnect_live():
    global client, client_task, translation_worker_task, translation_queue
    global connected, active_username, connection_status, last_error, last_comment_at

    async with connect_lock:
        old_client = client
        old_task = client_task
        old_worker = translation_worker_task
        client = None
        client_task = None
        translation_worker_task = None
        connected = False
        active_username = None
        connection_status = "idle"
        last_comment_at = None
        translation_queue = None

        if old_client is not None:
            try:
                await asyncio.wait_for(old_client.disconnect(), timeout=5)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)}"[:240]
        for task in (old_task, old_worker):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

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
