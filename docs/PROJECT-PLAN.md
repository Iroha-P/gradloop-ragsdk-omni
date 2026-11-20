# GradLoop 当前前端基线

2026-11-20 · Antigravity 版「研循·学术精工工作台」

## 审阅决定与验收记录

- **审阅决定**：2026-11-20 已批准推荐方案 A「**研循·学术精工工作台 (Editorial Split & Live Inspector)**」，结合实地浏览的 Awwwards SOTD（[Carter Ogunsola](https://www.awwwards.com/sites/carter-ogunsola)、[DeepBook](https://www.awwwards.com/sites/deepbook)、[Creative Reference Library](https://www.awwwards.com/sites/creative-reference-library)）与 Elements（[Console.log — Ethiack](https://www.awwwards.com/inspiration/console-log-ethiack)、[Colour-coded curriculum cards — CambriLearn](https://www.awwwards.com/inspiration/colour-coded-curriculum-cards-cambrilearn)、[Prompt Box — Brik.Space](https://www.awwwards.com/inspiration/prompt-box-brik-space)、[Edgeless Sidebar — Half the Sky](https://www.awwwards.com/inspiration/edgeless-sidebar-for-timeline-and-chat-half-the-sky)）完成 `app/ui/index.html` 改版。
- **构建与验证结果**：
  - 使用 `scripts/competition/build_public_site.py` 生成隔离公开预览（`dist/public-preview` 与本地实时预览路由 `http://127.0.0.1:61483/preview/`），本地离线备用演示 `http://127.0.0.1:61483/`（`competition-local/offline/index.html`）。面向公众的静态演示入口为 [GradLoop Demo](https://iroha-p.github.io/demo/gradloop/)。
  - 浏览器实测通过 `1440×900`（桌面）、`768×1024`（平板）、`390×844`（手机）三档视口走查，完整跑通「① 提问 → ② 查引用（点击胶囊高亮定位证据卡+内嵌摘要）→ ③ 练习批改（8/10 三维评分仪表+可折叠原始 JSON）→ ④ 错题重答（自动加载 `fixture-m-001` 并更新掌握度）」以及全模态文档模式确认闸门。
  - 已同步更新 `SHA256SUMS.json`，并通过 `scripts/public_release_scan.py` 源目录与发布包双重扫描。
