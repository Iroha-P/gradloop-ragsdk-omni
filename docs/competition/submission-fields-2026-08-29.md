# 创新应用赛道提交字段

## Demo 体验地址
https://gradloop-ragsdk-omni.pages.dev

## 演示视频链接
not_ready_until_user_uploads

## 应用场景分类
教育

## 项目说明

GradLoop RAGSDK Omni 是面向课程学习、技术面试与科研答辩训练的证据约束全模态 Agent。系统接收公开或合成的文字、图像、音频和视频材料，使用 MiniCPM-o 4.5 进行多模态理解，再通过 RAGSDK/BM25 检索带来源证据，交给 LangGraph 有界工作流生成结构化反馈、追问方向和下一步练习。

公开 Demo 通过 Worker 连接 MAP MiniCPM-o 4.5 Realtime；历史证据记录过文字 chat 的排队、首字和完整响应链路，但 2026-08-29 closeout 的 fresh smoke 为 `TimeoutError`，不以旧结果覆盖新鲜状态。同一协议为语音与视觉全双工交互预留入口，音频/视频浏览器验收尚未运行。离线模式使用仓库合成素材，验证检索、证据引用、反馈和行动确认闭环。在线 MAP 证据与 NPU 模型级 baseline 分开记录，公开环境不读取个人语料、私有文档、成绩证明或人工评分原件，请求级媒体不落盘。

## 开源仓库地址（选填）
https://github.com/Iroha-P/gradloop-ragsdk-omni

## 补充说明

- 在线文字 chat Smoke 已通过；Worker health 为 `ready/configured`。
- 音频/视频浏览器验收：not_run，提交时不要写成通过。
- 页面工程与 Worker 变量模板可复现；长期 MAP 凭据由参赛者在受控 Secret 存储中自行配置。
- 本地 v3 演示视频已验收；视频链接需由参赛者上传后替换。
- v3 PPTX 与同版 PDF 均已生成，并纳入 `innovation-final-v3` 提交包。
- 2026-08-30 closeout：脱敏候选通过双重扫描后以全新 Git 历史发布；开发仓库 remote 仍为空。
