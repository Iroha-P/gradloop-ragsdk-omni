<div align="center">

# GradLoop RAGSDK Omni

> 证据约束的全模态学习与面试训练 Agent

**🏆 星火杯 · 全球总决赛 20 强**

从学习材料到证据、从练习反馈到复盘，让 AI 辅导形成可检查、可恢复的学习闭环。

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![RAGSDK](https://img.shields.io/badge/RAG-RAGSDK-1677FF)](https://gitcode.com/Ascend/RAGSDK)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128)](LICENSE)
[![Privacy](https://img.shields.io/badge/Public%20Release-Privacy%20Scanned-18A058)](docs/release.md)

**[在线体验](https://gradloop-ragsdk-omni.pages.dev)** · **[功能亮点](#项目亮点)** · **[文档处理](#pdf--word-文档模式)** · **[快速开始](#quick-start)** · **[模型实验工程](https://github.com/Iroha-P/gradloop-ragsdk-omni-training)**

</div>

GradLoop 将可引用的 RAG 问答、有界 Agent 工作流、全模态理解和学习闭环组合在同一个可审计系统中。它面向课程学习、技术面试和科研答辩等场景，不只给出答案，还保留证据、记录薄弱项，并在需要改动学习计划时要求用户确认。

本仓库是 GradLoop 的**应用工程**：负责用户界面、API、检索、Agent 与学习状态；[Training 工程](https://github.com/Iroha-P/gradloop-ragsdk-omni-training)负责数据、模型实验、评测与推理服务。两仓属于同一项目，通过明确协议协作，而不是重复维护两套产品。

项目参加星火杯并取得**全球总决赛 20 强**成绩。该成绩属于 GradLoop 项目整体，不等同于任一模型评测指标或质量提升结论。

**[Online Demo](https://gradloop-ragsdk-omni.pages.dev)** · **[Architecture](docs/architecture.md)** · **[Reproducibility](docs/release.md)** · **[Demo Guide](docs/demo-script.md)**

![GradLoop 公开合成资料问答界面](docs/assets/demo-public-qa.png)

## 项目亮点

- **证据约束 RAG**：统一 BM25 baseline、RAGSDK 适配层和测试后端；回答携带引用、后端身份和脱敏 trace，证据不足时拒答。
- **有界 Agent 编排**：LangGraph 与显式 baseline 共用业务工具，限制工具步数和重试次数，对计划调整使用 human-in-the-loop 确认。
- **学习闭环**：把资料问答、学习计划、练习出题、答案批改、错题记录和重练串成可恢复的状态流程。
- **全模态扩展**：通过 MiniCPM-o 4.5 边界处理文本、图像、音频和视频帧；应用侧只接受明确确认的公开或合成媒体。
- **PDF / Word 文档模式**：PDF、DOC、DOCX 在浏览器本地提取正文并预览，用户检查并确认后才可把文本发送至 MAP；不上传原文件或原文件名。
- **隐私优先发布**：公开包仅包含源码、合成 fixtures、聚合指标和空配置示例；不包含私人语料、凭据、模型权重或原始评分记录。

## 公开 Demo

| 模块 | 公开页面行为 | 能力边界 |
| --- | --- | --- |
| 资料问答 | 公开合成回放 | 回答、引用与检索路径 |
| 全模态训练台 | MAP MiniCPM-o 4.5 Realtime 受控 API 入口 | 是否可用以当次会话为准；媒体只接受公开或合成内容 |
| PDF / Word 文档模式 | 源码已实现，在线页面待发布 | 浏览器本地提取正文；可选 MAP 文本分析需明确确认 |
| 学习计划 | 公开合成回放 | 结构化任务、时长与产出 |
| 练习批改 | 公开合成回放 | 出题、双评分与弱项归纳 |
| 错题本 | 公开合成回放 | 错题记录、重练与掌握度更新 |
| 知识库覆盖 | 公开合成回放 | 来源、章节、候选题与冲突统计 |

> 公开 Demo 不连接私人知识库。五个本地专属模块使用确定性合成回放，全模态训练台通过 Cloudflare Worker 代理访问 MAP，密钥不进入浏览器或仓库。

## PDF / Word 文档模式

文档模式已实现于本仓库并通过合成文件自动测试。2026-10-04 已修复本地页面缺少模块资源、文档误入媒体入口和离线状态说明不清的问题。Cloudflare 发布仍为 `prepared_not_final`：Wrangler 的 npx 缓存不完整，独立缓存重建在 60 秒限时内未完成，现有 Online Demo 尚未更新这一入口。Browser 运行依赖已可加载，但保存权限校验未通过，真实浏览器验收仍未完成。

在全模态训练台选择 **“文档模式 · PDF / Word”**：

1. 选择公开或合成 PDF、DOC 或 DOCX，点击 **“解析文档（仅本机）”**。在媒体入口选择文档也会自动转入文档模式；文档与媒体不能混选。
2. 查看实际提取的正文、页数或段落数；可在预览中修改文本。
3. 检查文本为公开或合成内容，再勾选确认，选择是否发送至 MAP 分析。
4. 结果须对照原文复核。点击 **“清除文档”** 可清除本次浏览器内存内容。

| 格式 | 处理能力 | 明确限制 |
| --- | --- | --- |
| PDF | 提取文字层并保留页码标记 | 最多 50 页；扫描件需先 OCR，不自动识别图片或还原排版 |
| DOCX | 提取 Word 正文文字 | 不执行宏、嵌入对象或外部链接；不还原复杂排版 |
| DOC | 解析 OLE 格式旧版 Word 正文 | 损坏、加密或其他伪装成 `.doc` 的格式须先转换 |

最多 4 份文件，每份 ≤ 10 MB、合计 ≤ 20 MB，合并正文 ≤ 12,000 字符。超限不静默截断。解析器与许可证随站点托管，不把文档发送给第三方 CDN。文档文本分析不等于已将资料加入 RAG 知识库。

详见 [文档模式与安全边界](docs/document-mode.md)。

## 系统架构

```mermaid
flowchart LR
    U[Web / API] --> S[安全与隐私闸门]
    S --> A[LangGraph Learning Agent]
    A --> T[计划 / 题库 / 批改 / 错题]
    A --> R[RAG Pipeline]
    R --> B{Backend Factory}
    B --> BM[BM25 Baseline]
    B --> RS[RAGSDK Adapter]
    R --> E[引用与证据门]
    A --> O[脱敏 Trace / Metrics]
    S --> M[MiniCPM-o Multimodal Boundary]
```

RAG 后端在启动时显式选择。当 `ragsdk` 不可用时，系统返回 `backend_unavailable`，不会静默伪装成 RAGSDK 运行。

## Quick Start

需要 Python 3.11 或 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\public_release_scan.py
.\.venv\Scripts\python.exe scripts\run_public_demo.py
```

打开 <http://127.0.0.1:8000/>。公开启动器只绑定 loopback，并强制使用版本化合成资料和题库，不读取现有私人存储。

该启动方式默认是 **BM25 + LangGraph 离线基线**，不是模型 API 运行模式。PDF / Word 的正文提取不需要模型，也不上传原文件；只有另行配置受允许的 MAP 代理、检查预览并确认发送后，才会发起模型分析。在线 Pages 与本地服务是两个独立入口，Worker 的 `upstream=configured` 也不代表某次模型推理已经成功。

已在运行的本地服务更新代码后，需要在原启动窗口按 `Ctrl+C`，重新运行上述启动命令，再在页面按 `Ctrl+F5`。只刷新网页不会加载新增的后端资源路由；`ERR_CONNECTION_REFUSED` 则说明本机服务没有在该端口接受连接。

运行公开组合评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_public_portfolio.py
```

或启动隔离的 CPU 容器基线：

```powershell
docker compose --profile cpu up --build
```

启动后检查 <http://127.0.0.1:8000/health> 和 <http://127.0.0.1:8000/ready>。容器以非 root 用户运行，只发布到 `127.0.0.1:8000`，包含 20 份合成文档和 5 道合成练习题。

## 已验证证据

| 范围 | 版本化结果 | 诚实边界 |
| --- | --- | --- |
| CPU 公开基线 | Docker health `healthy`，20 份合成文档，Agent SSE 链路通过 | BM25 + LangGraph，不等于 RAGSDK runtime |
| MiniCPM-o 4.5 / Ascend | 文图、音频、图音、视频帧四路 smoke 通过 | 公开模型素材和聚合运行证据 |
| 应用 E2E | LangGraph + MiniCPM-o Ascend 公开图文请求通过 | 请求级保留，不写入输入内容 |
| 人工双评 | 2 名评分者、30 条五维评分；QWK `0.5645`，Pearson `0.7331` | 衡量量表一致性，不宣称模型质量提升 |

详细报告位于 [`reports/public/`](reports/public/)。公开合成评测不代表真实用户准确率，可观测指标也不代替语义评审。

## 目录导航

```text
app/                 FastAPI、Agent、RAG 与学习工具
configs/             公开可复现配置
data/public_demo/    合成演示资料与题库
deploy/              MAP Realtime Worker 公开配置
docs/                架构、部署、隐私与演示文档
reports/public/      脱敏聚合评测证据
scripts/             发布、评测与安全扫描工具
tests/               单元、集成、隐私与发布契约测试
```

## 隐私与发布边界

- 不提交个人文档、成绩证明、身份信息、原始评分或本地知识库。
- 不提交 API Key、Cookie、绝对路径、模型权重或第三方受限内容。
- RAGSDK 是独立配置的外部依赖，本仓库不重新分发其源码。
- 构建公开包时只读取 [`release-manifest.json`](release-manifest.json) 白名单，生成 SHA-256 清单并对产物再扫描。

## 文档

- [系统架构](docs/architecture.md)
- [发布与可复现指南](docs/release.md)
- [MiniCPM-o 全模态集成](docs/minicpmo-integration.md)
- [MAP Realtime 部署](docs/competition/deployment.md)
- [MAP 一键配置与部署](docs/competition/one-key-map-setup.md)
- [PDF / Word 文档模式](docs/document-mode.md)
- [2–4 分钟演示脚本](docs/demo-script.md)
- [故障排查](docs/troubleshooting.md)
- [开源许可](LICENSE)
- [第三方归属与 RAGSDK 边界](NOTICE)

## 当前状态

公开 Demo、CPU baseline、发布扫描和公开聚合评测均可复现。RAGSDK 完整 runtime 需在满足 Linux、Milvus 和模型服务依赖的环境中单独配置；系统不会把 baseline 结果宣称为 RAGSDK 实测结果。

Apache-2.0 licensed. See [NOTICE](NOTICE) for attribution and external dependency boundaries.
