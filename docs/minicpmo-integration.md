# MiniCPM-o 全模态应用接入

本项目把 MiniCPM-o 作为可选的外部全模态推理服务接入 GradLoop。应用仓库不包含模型权重、训练语料、私人资料或云端凭证；未配置服务时，原有 BM25、RAGSDK 适配与学习闭环仍可独立运行。

## 已实现的应用边界

- `GET /ready` 报告全模态后端、模型、支持模态和真实可用状态；服务不可达时不会伪装成功。
- `POST /v1/omni/analyze` 接受最多 4 个图像、音频或视频文件，并要求调用方显式确认 `content_policy=public_or_synthetic`。
- 服务端按 MIME 类型、文件头、数量和大小做白名单校验；拒绝路径式文件名和伪造格式。
- 转发给模型服务的载荷只有内容寻址 ID、媒体类型和字节内容，不发送原始文件名、本地路径或用户标识。
- 文件只在当前请求范围处理，不写入知识库、SQLite、应用日志或学习记录。Web 框架可能把较大 multipart 分片短暂写入系统临时区，关闭请求后由框架清理，因此准确承诺是“不做应用持久化”，不是“纯内存”。
- 远端地址必须使用 HTTPS；只有 `127.0.0.1`、`localhost` 和 `::1` 允许 HTTP。目标主机还必须出现在显式白名单中。
- API token 只从进程环境变量读取，不进入配置对象、`/ready` 响应或版本控制。

## 本地应用配置

先安装项目依赖，然后在当前终端设置：

```powershell
$env:BAOYAN_MULTIMODAL_BACKEND="minicpmo"
$env:BAOYAN_MINICPMO_BASE_URL="http://127.0.0.1:18080"
$env:BAOYAN_MINICPMO_ALLOWED_HOSTS="127.0.0.1,localhost,::1"
$env:BAOYAN_MINICPMO_MODEL="openbmb/MiniCPM-o-4_5"
```

若通过 SSH 隧道把昇腾服务映射到本机 `127.0.0.1:18080`，不需要在应用中保存云端地址或凭证。若直接访问远端服务，必须使用 HTTPS，并把远端域名加入 `BAOYAN_MINICPMO_ALLOWED_HOSTS`；认证值通过 `BAOYAN_MINICPMO_API_TOKEN` 临时注入。

启动后检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

只有 `multimodal.ready=true` 时，网页的“全模态训练台”才代表真实模型已连接。

## 模型服务契约

应用期望外部服务提供两个端点：

1. `GET /health`
2. `POST /v1/omni/generate`

生成请求的最小结构如下：

```json
{
  "model": "openbmb/MiniCPM-o-4_5",
  "prompt": "请分析公开合成材料",
  "attachments": [
    {
      "asset_id": "sha256-prefix",
      "media_type": "image/png",
      "content_base64": "..."
    }
  ],
  "max_tokens": 512,
  "stream": false
}
```

生成响应至少应包含 `content`；推荐同时返回 `model`。健康检查应返回 `ready`、`model` 和 `modalities`。应用会拒绝超过 2 MiB 的异常响应体。

## 昇腾侧仍需完成

本次应用接入不等于 MiniCPM-o 已在 910C 上完成推理或训练。提交真实成果前还需在独立训练仓库完成：

1. 固定 ModelScope 模型 revision、CANN、PyTorch、torch_npu 与 Python 版本。
2. 在 910C 上跑通官方模型最小推理，并记录峰值显存、首 token 延迟、总时延和错误恢复。
3. 用公开或合成输入实现上述 HTTP 契约，不上传本地私人语料。
4. 通过 SSH 隧道或受控 HTTPS 连接本应用，验证图像、音频、视频各至少一个端到端案例。
5. 再决定是否做 LoRA。应用联调和 baseline 不要求训练；只有公开数据、任务指标和冻结测试集定义完成后才进入微调。

## 安全验收

- 私人资料、成绩证明、证件、真实简历、聊天记录和私人知识库禁止进入全模态接口。
- 模型服务日志不得记录请求正文、媒体字节、token、绝对路径或用户身份。
- 公开展示固定使用版本化公开数据或合成数据，并由发布扫描器复验。
- 任何云端实验只记录聚合指标和匿名运行信息；模型权重与缓存不进入应用仓库。
