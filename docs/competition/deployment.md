# 部署说明（公开 Demo）

## 已发布地址

- Pages：<https://gradloop-ragsdk-omni.pages.dev>
- Worker health：`GET https://gradloop-map-realtime-proxy.minicpmo-demo-integration.workers.dev/health`
- Worker WebSocket：`wss://gradloop-map-realtime-proxy.minicpmo-demo-integration.workers.dev/ws?mode=chat`

Pages 只承载公开合成素材和前端代码；Worker 只在当前 WebSocket 会话中转发请求。MAP API Key 仅以 Cloudflare Worker Secret 名称 `MAP_API_KEY` 存在，变量值不进入仓库、日志、报告或浏览器。

## 本地公开回放

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/public_release_scan.py --root .
.\.venv\Scripts\python.exe scripts/run_public_demo.py
```

公开回放只使用 `data/public_eval/` 与 `data/public_demo/`，不扫描或读取本地私人资料。静态发布目录由 `scripts/competition/build_public_site.py` 生成，最终目录形如 `dist/competition/one-key-pages/final-*`。

## MAP 代理配置（仅 Cloudflare 侧）

非敏感变量：`ALLOWED_ORIGINS`、`MAP_REALTIME_URL`、`MAP_AUTH_HEADER`、`MAP_AUTH_PREFIX`、`MAP_SESSION_SECONDS`。当前默认上游为 `wss://minicpmo45.modelbest.cn/v1/realtime`，Worker 内部按 Cloudflare Upgrade fetch 规范转换为 HTTPS 子请求。长期密钥只通过 Wrangler 隐藏输入框配置：

```powershell
npx wrangler secret put MAP_API_KEY --config .\deploy\map-realtime-worker\wrangler.one-key.jsonc --name gradloop-map-realtime-proxy
```

不要在聊天、命令参数、`.env`、截图或提交材料中输入或显示 Key。

## 故障恢复

- `403 origin_not_allowed`：只允许已发布 Pages 精确来源，不要放宽白名单。
- `503 proxy_not_configured`：检查 Secret 名称是否存在，不读取 Secret 值。
- `502 upstream_unavailable`：MAP 上游或鉴权暂不可用，记录为失败并稍后重试。

真实在线结果须来自本轮运行；音频/视频没有浏览器实测时保持 `not_run`。昇腾模型 revision、设备和性能指标只引用独立 NPU 证据文件，不能手工补写。
