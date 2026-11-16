"""
FastAPI 主程序

完整流程：用户输入中文 → 情感识别 → 风险评估 → 大模型干预回复

配置：将 .env.example 复制为 .env，填入 API Key，项目启动时自动加载。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 最先加载配置
from configs import load_config
load_config()

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import emotion, risk, chat, federated, multimodal, auth, survey, admin_mgmt
from .utils.conversations import get_recent_conversations_filtered, get_risk_stats
from .utils.auth import get_current_user
from .utils.rbac import is_any_admin, normalize_role
from .utils.db import init_tables

app = FastAPI(
    title="百川——校园智心守护系统 API",
    description="情感分析、风险评估、AI 干预",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(emotion.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(federated.router, prefix="/api/v1")
app.include_router(multimodal.router, prefix="/api/v1")
app.include_router(survey.router, prefix="/api/v1")
app.include_router(admin_mgmt.router, prefix="/api/v1")

init_tables()


# API 信息（部署后前端与 API 同源，根路径用于提供前端页面）
@app.get("/info")
def root():
    return {
        "name": "百川——校园智心守护系统",
        "apis": [
            "POST /api/v1/emotion/analyze",
            "POST /api/v1/risk/assess",
            "POST /api/v1/chat/message",
            "GET /api/v1/chat/history",
            "POST /api/v1/multimodal/analyze",
            "GET /api/v1/federated/status",
        ],
    }


@app.get("/api/v1/admin/conversations")
def admin_conversations(request: Request, limit: int = 200):
    """管理员：按租户查看对话（校级=本校+公共池；超级=全量）。需 Bearer token。"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "需要登录")
    if not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    role = normalize_role(user.get("role"))
    school = (user.get("school") or "").strip()
    items = get_recent_conversations_filtered(
        limit=limit, viewer_role=role, viewer_school_id=school,
    )
    return {"items": items}


@app.get("/api/v1/admin/risk-stats")
def admin_risk_stats(request: Request):
    """管理端：风险等级统计（凸显预警/危机规模）。"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "需要登录")
    if not is_any_admin(user.get("role")):
        raise HTTPException(403, "需要管理员权限")
    role = normalize_role(user.get("role"))
    school = (user.get("school") or "").strip()
    stats = get_risk_stats(viewer_role=role, viewer_school_id=school)
    return stats


# 挂载前端静态文件，使部署后用户通过同一域名访问网页（/ 为 index.html，/admin.html 为管理端）
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
