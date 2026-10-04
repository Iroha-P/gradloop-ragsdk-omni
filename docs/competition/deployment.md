# 部署说明（公开 Demo）

一键配置、确认步骤与恢复方式见 [MAP 一键配置说明](one-key-map-setup.md)。

2026-10-05 的 [PDF / Word 文档模式](../document-mode.md)与 MAP 前端修复尚未发布到 Pages，状态为 `prepared_not_final`。已独立安装并验证 Wrangler 4.76.0，但 Cloudflare 身份请求未成功。下面的已发布地址仍指向上一版页面，不代表本次功能已上线。Browser 保存权限校验失败，浏览器实测与本轮模型推理尚未验证；此前的缓存失败记录保留，不当作本轮结果。

静态 Pages 不运行本地 FastAPI：媒体文件分析按钮明确禁用，PDF / Word 在浏览器提取正文，确认后的正文与公开文本分析使用 MAP WebSocket 代理。前端不得请求静态站点的本地 `/v1/` 接口；模型成功状态必须来自非空文字回复及完成事件。音频与视频实时输入在完成实现和验收前保持未开放，不申请设备权限。

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

该启动器默认设置生成后端为 `none`，属于 BM25 + LangGraph 离线基线，不会自动调用模型 API。文档解析在浏览器本地完成；模型分析另需允许的代理配置与用户确认。Worker health 的 `upstream=configured` 只说明上游已配置，不证明本轮模型调用成功。

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
