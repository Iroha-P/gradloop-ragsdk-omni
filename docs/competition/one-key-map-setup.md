# GradLoop RAGSDK Omni：一键配置 MAP 实时 Demo

本页只说明公开 Demo 的部署。它不会读取本地保研资料、飞书私有内容、成绩证明、证件或人工评分原件；Pages 只承载公开合成案例，Worker 只转发当前会话。

## 你只需要做三件事

1. 在本项目 worktree 根目录运行下面的一条命令：

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/competition/setup_map_demo.ps1
   ```

2. 如果浏览器打开 Cloudflare 授权页，按页面提示完成授权并返回终端。
3. 看到 Wrangler 自己的隐藏输入框后，只在该输入框粘贴 MAP API Key；不要把 Key 发给任何人、放进聊天、截图、Git、`.env` 或命令参数。

脚本会先完成本地测试、Ruff、Node 测试、公开发布扫描和静态 Pages 构建。随后它会显示 Worker 名称、Pages 项目名和 MAP 主机，并要求输入精确的 `DEPLOY`。输入其他内容会安全退出，不执行外部操作。

## 成功时应看到什么

- Worker：一个公开的 HTTPS 地址；浏览器实际使用其对应的 `wss://.../ws` 地址。
- Pages：`https://gradloop-ragsdk-omni.pages.dev`。
- 健康检查：`ready` 且上游为 `configured`。
- chat 脱敏 Smoke：`passed`。

以上只代表文字 chat 链路已通过。audio/video 只有在使用公开合成素材单独验证后才能写成通过；没有实测时必须保留为 `not_run`。

## API Key 的安全边界

- Key 由 Wrangler 的隐藏提示框直接写入 Cloudflare Worker Secret；向导不读取、不打印、不保存 Key。
- 不要在 PowerShell 参数、系统环境变量、`.env`、`.dev.vars`、JSON、日志、截图、PPT、PDF 或视频中保存 Key。
- 本地报告只保存公开 URL、状态、时间和关闭原因，不保存提示词、回答、音频、图像、请求头或完整查询字符串。

## 失败恢复

向导遇到失败会立即停止，不会自动清理已有部署。请在 Cloudflare 控制台中按以下位置人工复核：

- Worker Secret：`Worker → Settings → Variables and Secrets → MAP_API_KEY → Delete`。
- Worker 或 Pages 项目：在对应项目的设置页确认名称后，按控制台提供的删除/暂停入口人工操作。

不要把账号 ID、Key 或控制台截图提交到仓库。仓库只保留本地脚本和公开配置模板。

## 常见问题

| 现象 | 含义 | 处理 |
|---|---|---|
| `preflight failed` | 本地依赖、测试、扫描或 worktree 检查未通过 | 先修复终端中显示的步骤；不要输入 `DEPLOY` |
| Cloudflare 授权失败 | Wrangler 没有获得当前账号授权 | 重新运行脚本，在浏览器完成授权；不要把授权信息发到聊天 |
| `403 origin_not_allowed` | 页面来源不在 Worker 白名单 | 确认使用脚本生成的 Pages 地址；不要手动放宽白名单 |
| `503 proxy_not_configured` | Worker 尚未成功写入 Secret 或公开变量 | 在 Wrangler 隐藏框重新设置 Secret，检查 Worker 名称 |
| `502 upstream_unavailable` | MAP 上游暂时不可用或协议不兼容 | 稍后重试；记录为失败，不把它写成实时通过 |
| Pages 无法访问 | Pages 项目创建或发布失败 | 检查 Cloudflare 项目状态和公开地址；保留 Worker，不提交虚构链接 |
| chat Smoke 失败 | WebSocket、来源或上游响应未完成 | 只记录脱敏错误类别；audio/video 不得顺带宣称通过 |

## 输出位置

- 本地验证 Pages 目录：`dist/competition/one-key-pages/validation-*`。
- 最终 Pages 目录：`dist/competition/one-key-pages/final-*`。
- 本地脱敏摘要：`reports/local/competition/one-key-map-latest.json`。

这些运行输出默认被 Git 忽略，不应上传。公开仓库只包含脚本、无密钥配置和本说明。
