# GradLoop 应用工程指南

GradLoop RAGSDK Omni 面向课程学习、技术面试与科研答辩，提供带引用问答、学习计划、练习批改、错题复习、有界 Agent 工作流和全模态接入边界。

## 项目分工

| 工程 | 负责什么 |
| --- | --- |
| [本应用仓库](https://github.com/Iroha-P/gradloop-ragsdk-omni) | RAG、Agent、学习工具、API、UI、应用评测与公开 Demo |
| [Training 仓库](https://github.com/Iroha-P/gradloop-ragsdk-omni-training) | 数据准备、模型实验、冻结评测、NPU Base/LoRA 与模型服务 |

应用与训练通过明确的 API 和数据协议连接。各自保留独立评测和版本记录，不复制私人资料或模型权重。

## 目录和阅读入口

- `app`：API、Agent、RAG、学习工具和前端。
- `data/public_demo` 与 `data/public_eval*`：版本化公开合成资料。
- `deploy`：MAP Realtime 代理的公开代码与配置。
- `reports/public`：历史聚合证据；不是实时可用性保证。
- [系统架构](./docs/architecture.md)、[发布指南](./docs/release.md)、[演示说明](./docs/demo-script.md)。

## 开发与维护

先阅读 README 和对应模块测试，再修改功能。应用功能在本仓库实现，训练或模型服务功能在 Training 仓库实现。

RAGSDK 适配代码、合约测试和真实 RAGSDK 环境验证分别记录。MiniCPM-o 运行证据、合成回放和当前在线体验也分别记录；未实测的结果不能写成已通过。

## 公开与归档边界

本仓库是独立脱敏发布版本，只保留选定源码、合成数据、文档和聚合证据。它不是本地开发仓库全部历史或私人学习状态的备份。

源码、协议、冻结评测、原始实验与发布证据保留版本。清理只针对已确认可重建的缓存和临时构建，先核实精确路径及唯一产物。发布时按 manifest 构建并检查源文件与成品，禁止上传私人语料、凭据、数据库和模型权重。
