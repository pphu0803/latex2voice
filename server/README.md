# latex2voice trial server

免费试用服务骨架：Google OAuth 登录 + 每人每 ISO 周 5 次转换。

## 本地运行

```bash
export GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
export GOOGLE_CLIENT_SECRET=xxx
export L2V_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
export L2V_REDIRECT=http://localhost:8080/auth/callback   # 生产换 https://api.latex2voice.dev/auth/callback

H:/SHARE/latex-tts/venv/Scripts/python.exe -m uvicorn server.main:app --port 8080   # repo 根目录下
```

## Google OAuth 配置（Google Cloud Console）

1. API & Services → OAuth consent screen：External，填 app 名/支持邮箱
2. Credentials → Create OAuth client ID → Web application
3. Authorized redirect URI 填 `L2V_REDIRECT` 的值（本地调试 `http://localhost:8080/auth/callback`，
   生产 `https://api.latex2voice.dev/auth/callback`）

## 设计说明

- **额度**：`usage` 表按 ISO 周键（`2026-W47`）计数，自然周一 UTC 重置；无订阅时这是防刷的唯一闸门，
  上生产后建议加 IP/设备指纹辅助限流
- **模型加载**：首请求懒加载 TN + Kokoro，单 worker 进程内常驻；GPU/多 worker 部署时把
  `reader._get_model()` 换成独立推理服务
- **缓存**：音频按 spoken 文本 sha1 缓存（同公式不重算）
- **升级路径**：SQLite → Postgres（只换 `db.py`）；试用 → credits（usage 表加 `kind` 已预留）；
  PDF 上传 → 任务队列 + 对象存储（R2）

## API

| 路由 | 说明 |
|---|---|
| `GET /auth/login` → `/auth/callback` | Google OAuth |
| `GET /api/me` | 邮箱 + 本周已用/额度 |
| `POST /api/trial {text}` | 消耗 1 次，返回 `{spoken, audio_url, remaining}`；2000 字符上限 |
| `GET /healthz` | 探活 |
