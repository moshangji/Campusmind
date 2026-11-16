"""联邦学习相关接口（预留）"""

from fastapi import APIRouter

router = APIRouter(prefix="/federated", tags=["federated"])


@router.get("/status")
def federated_status():
    """联邦学习训练状态（预留）"""
    return {"status": "idle", "message": "联邦学习接口待实现"}
