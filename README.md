# 百川——联邦协同模式下的校园智心守护系统

校园心理健康辅助系统。运行主体在 `bert-master/`。

## 快速启动

```powershell
cd bert-master
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env`，填入本次启用的接口与数据库（不要把填好的 `.env` 提交到仓库）。然后：

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/`（学生端）或 `/admin.html`（管理端）。`/info` 可确认服务已启动。

情感模型权重体积较大，未放入本仓库。请将 `final_model` 放到 `bert-master/output/` 对应目录后再做完整演示。未配置大模型或数据库时，匿名文字风险主链仍可按部署说明使用。
