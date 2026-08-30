# 项目说明（创新应用赛道）

GradLoop RAGSDK Omni 是面向课程学习、技术面试与科研答辩训练的证据约束全模态 Agent。用户可以提交公开或合成的图像、音频、视频与文字材料，系统先做格式和内容政策校验，再由 MiniCPM-o 4.5 进行多模态理解；RAGSDK/BM25 返回带来源的证据片段，LangGraph 以有界状态流生成结构化反馈、追问方向和下一步练习。

项目将在线能力与本地可复现能力分层：公开 Pages Demo 通过 Cloudflare Worker 代理 MAP MiniCPM-o 4.5 Realtime，支持文字 chat，并为语音、视觉全双工交互保留同一协议入口；本轮真实在线验收已确认 chat 链路，音频/视频浏览器验收仍标记为未运行。离线模式使用仓库内合成素材，验证 RAGSDK 检索、证据引用、反馈和行动确认的完整闭环。

核心工程价值是可审计和可复现：前端不持有长期密钥，Worker Secret 不写入仓库；公开 Demo 不读取个人保研资料、飞书私有文档、成绩证明或人工评分原件；请求级媒体不落盘。在线 MAP 结果与昇腾 NPU 模型级 baseline 分开记录，避免用离线或历史指标替代真实在线效果。

公开 Demo：<https://gradloop-ragsdk-omni.pages.dev>

本地复现和发布边界见 [部署说明](deployment.md) 与根目录 [PROJECT_DESCRIPTION.md](../../PROJECT_DESCRIPTION.md)。
