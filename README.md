# TK Live Translator Bridge

FastAPI backend for testing public TikTok LIVE comment collection and translation into Simplified Chinese.

## Current endpoints
- `GET /` - service overview
- `GET /api/health` - health and connection state
- `GET /api/live/status` - TikTok listener status
- `POST /api/live/connect` - connect to a public creator using their TikTok @username
- `GET /api/live/comments` - latest 500 comments and translation status
- `POST /api/live/disconnect` - disconnect the listener
- `POST /api/translate/test` - test translation independently

## Translation
The backend detects the source language locally and uses the MyMemory translation API to translate non-Chinese comments into Simplified Chinese. Chinese comments are passed through unchanged. MyMemory is a limited free service; anonymous usage is limited to 5,000 characters/day. Set the optional `MYMEMORY_EMAIL` environment variable in Render to raise the daily limit to 50,000 characters/day. Provider failures are now returned in `translation_error` rather than silently hidden.

## Deploy
Render should build from the repository root using the root `Dockerfile`. After pushing an update, use Render's Manual Deploy > Deploy latest commit if automatic deploy is disabled.

## Important limitations
- TikTokLive is an unofficial reverse-engineered integration. TikTok may change the protocol, signing service, and regional availability. It is suitable for a test, not a guaranteed production connector.
- The current backend does not implement the separate feature of translating the host's Chinese speech into spoken English.
- This test backend has no access-code authentication and currently uses permissive CORS. Do not expose it as a public production service or put credentials in requests.
- Do not enter TikTok passwords, session cookies, or private account credentials.
- Review the exact TikTokLive license and data-processing terms before company-wide use.
