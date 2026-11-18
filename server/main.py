"""latex2voice trial service: Google OAuth + 5 conversions per ISO week.

Env vars required:
  L2V_SECRET      random string, signs the session cookie
  GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET   (Google Cloud Console -> OAuth consent)
  L2V_REDIRECT    external URL of /auth/callback, e.g. https://api.latex2voice.dev/auth/callback
  L2V_TRIALS_PER_WEEK   default 5

Run:  uvicorn server.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from authlib.integrations.starlette_client import OAuth  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse  # noqa: E402
from itsdangerous import BadSignature, URLSafeSerializer  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.middleware.sessions import SessionMiddleware  # noqa: E402

sys.path.insert(0, ".")
from server import db  # noqa: E402

SECRET = os.environ.get("L2V_SECRET", "dev-secret-change-me")
TRIALS_PER_WEEK = int(os.environ.get("L2V_TRIALS_PER_WEEK", "5"))
COOKIE = "l2v_session"

app = FastAPI(title="latex2voice trial")
app.add_middleware(SessionMiddleware, secret_key=SECRET, same_site="lax", https_only=True)

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

ser = URLSafeSerializer(SECRET, salt="l2v-session")


def current_user(request: Request) -> dict | None:
    tok = request.cookies.get(COOKIE)
    if not tok:
        return None
    try:
        sub = ser.loads(tok)
    except BadSignature:
        return None
    row = db.get_user(sub)
    return {"sub": sub, "id": row[0], "email": row[1], "name": row[2]} if row else None


def require_user(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "sign in at /auth/login")
    return u


# ---------- auth ----------

@app.get("/auth/login")
async def auth_login(request: Request):
    redirect = os.environ.get("L2V_REDIRECT", request.url.replace(path="/auth/callback"))
    return await oauth.google.authorize_redirect(request, str(redirect))


@app.get("/auth/callback")
async def auth_callback(request: Request):
    tok = await oauth.google.authorize_access_token(request)
    info = tok["userinfo"]
    uid = db.upsert_user(info["sub"], info.get("email", ""), info.get("name", ""))
    resp = RedirectResponse("/")
    resp.set_cookie(COOKIE, ser.dumps(info["sub"]), httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie(COOKIE)
    return resp


# ---------- quota + trial ----------

@app.get("/api/me")
def me(u: dict = Depends(require_user)):
    return {"email": u["email"], "name": u["name"], "used": db.used_this_week(u["id"]),
            "limit": TRIALS_PER_WEEK, "week": db.week_key()}


class TrialBody(BaseModel):
    text: str  # plain text with optional $...$ LaTeX; full PDF reader comes later


@app.post("/api/trial")
def trial(body: TrialBody, u: dict = Depends(require_user)):
    if len(body.text) > 2000:
        raise HTTPException(413, "trial text limit is 2000 chars")
    used = db.used_this_week(u["id"])
    if used >= TRIALS_PER_WEEK:
        raise HTTPException(429, f"weekly limit reached ({TRIALS_PER_WEEK}/week). Next reset: Monday.")
    # heavy import on first use only; single-worker process
    from src.latextts.reader import convert_text  # noqa: PLC0415  (prototype; wired when reader lands)
    spoken, audio_path = convert_text(body.text)
    db.record_use(u["id"])
    return JSONResponse({"spoken": spoken, "audio_url": audio_path,
                         "remaining": TRIALS_PER_WEEK - used - 1})


# ---------- minimal UI ----------

PAGE = """<!doctype html><meta charset=utf-8><title>latex2voice trial</title>
<body style="font-family:Georgia,serif;max-width:640px;margin:60px auto;color:#1f2933">
<h1>latex2voice <span style=color:#8c1d18>trial</span></h1>
{body}
</body>"""


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    u = current_user(request)
    if not u:
        return PAGE.format(body='<a href="/auth/login" style="background:#1f2933;color:#fff;padding:10px 18px;'
                            'border-radius:8px;text-decoration:none">Sign in with Google</a>')
    left = TRIALS_PER_WEEK - db.used_this_week(u["id"])
    return PAGE.format(body=f"""
<p>Signed in as {u['email']} · <b>{left}</b> / {TRIALS_PER_WEEK} trials left this week ·
<a href="/logout">sign out</a></p>
<textarea id=t rows=6 style="width:100%" placeholder="Paste text with $...$ LaTeX formulas…"></textarea><br>
<button onclick="go()" style="margin-top:8px;background:#8c1d18;color:#fff;border:0;padding:10px 22px;border-radius:8px;cursor:pointer">Convert</button>
<pre id=out style="white-space:pre-wrap;background:#faf7f2;padding:12px;border-radius:8px"></pre>
<script>
async function go() {{
  const r = await fetch('/api/trial', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{text: document.getElementById('t').value}})}});
  const j = await r.json();
  document.getElementById('out').textContent = r.ok ? (j.spoken + '\\n\\n[audio: ' + j.audio_url + ']') : (j.detail || r.status);
}}
</script>""")


@app.get("/healthz")
def healthz():
    return {"ok": True}
