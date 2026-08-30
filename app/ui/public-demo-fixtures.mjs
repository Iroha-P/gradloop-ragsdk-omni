const ACTIONS = Object.freeze(["ask", "plan", "question", "grade", "mistakes", "knowledge"]);

const FIXTURES = Object.freeze({
  ask: {
    answer: "公开合成回放：先用结论—方法—证据组织自我介绍，再把追问引向已经准备好的技术细节。",
    backend: "public_fixture",
    retrieval: { candidate_count: 4, selected_count: 2, elapsed_ms: 12 },
    citations: [
      { title: "公开合成·表达结构示例", section: "证据约束", chunk_id: "fixture-ask-01" },
      { title: "公开合成·追问引导示例", section: "面试训练", chunk_id: "fixture-ask-02" }
    ]
  },
  plan: {
    tasks: [
      { day: 1, title: "搭建回答骨架", minutes: 35, deliverable: "完成一版结论—方法—证据提纲" },
      { day: 2, title: "复述技术细节", minutes: 40, deliverable: "录制一次两分钟结构化回答" },
      { day: 3, title: "模拟追问复盘", minutes: 30, deliverable: "整理三个可验证的追问分支" }
    ]
  },
  question: {
    questions: [{ question_id: "fixture-q-001", question: "请用结论、方法和证据说明一个你熟悉的技术方案。" }],
    warnings: []
  },
  grade: {
    engine: "Public Fixture Workflow",
    status: "completed",
    current_node: "review",
    next_nodes: [],
    steps: ["结构识别", "证据检查", "表达建议"],
    recovery: { resumed: false },
    result: {
      baseline_score: 8,
      dimensions: { structure: 3, evidence: 3, reflection: 2 },
      missing_points: ["补充一个可复核的结果"] ,
      improved_answer: "先给出结论，再解释方法，最后用一个可复核结果收束。",
      next_action: "重答一次并控制在两分钟内。",
      warnings: ["公开合成回放，不是实时本地后端结果"]
    }
  },
  mistakes: {
    mistakes: [{
      mistake_id: "fixture-m-001",
      question_id: "fixture-q-001",
      mastery_status: "reviewing",
      score: 6,
      latest_score: 6,
      redo_count: 0,
      consecutive_passes: 0,
      next_review_date: "公开回放 +2 天",
      weak_tags: ["证据闭环"]
    }]
  },
  knowledge: {
    status: "ready",
    coverage: {
      source_count: 3,
      section_count: 8,
      candidate_count: 24,
      canonical_count: 18,
      duplicate_group_count: 2,
      conflict_group_count: 0,
      generic_count: 18,
      personal_count: 0,
      unresolved_count: 0,
      sources: [{ status: "public_synthetic" }, { status: "public_synthetic" }, { status: "public_synthetic" }]
    }
  }
});

export function isPublicFixtureAction(action) {
  return ACTIONS.includes(String(action));
}

export function getPublicFixture(action, _input = {}) {
  if (!isPublicFixtureAction(action)) {
    throw new Error(`未知公开回放动作：${String(action)}`);
  }
  return structuredClone(FIXTURES[action]);
}
