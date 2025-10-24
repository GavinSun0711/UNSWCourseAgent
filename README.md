# UNSW Course Agent

一个基于 **LangGraph** 的课程问答与规划 Chatbot，面向 **UNSW 课程信息查询、两年AI方向学习规划、轻量口碑RAG**，并支持 **CSV / ICS 导出**。提供 **CLI** 与 **Gradio Web UI** 两种入口。

> 适合提交 GitHub 的课程/作业项目：结构清晰、可本地启动、含接口与工程规范说明。

---

## ✨ 核心功能
- **课程信息查询**：学期（T1/T2/T3）、先修/互斥、类别、课程简介。
- **两年AI方向学习规划**（6个Term）：考虑类别配额、先修约束、学期负荷、Project/Research 偏好。
- **按需解释**：生成规划后输入“**请给解释**”，返回逐课理由与数据依据。
- **本地 Reviews RAG**：从 `course_reviews.jsonl` 汇总评分/难度/工作量与高频优缺点，附简短引用。
- **导出**：
  - `plan.csv`：`Term, CourseCode, CourseName, Description`
  - `plan.ics`：每门课一个全天事件（可导入 Google/Apple/Outlook）。
- **双入口**：CLI / Gradio Web UI。

---

## 🚀 快速开始

```bash
# 1) 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 配置环境变量（不要提交密钥）
cp .env.example .env
# 打开 .env 设置：DASHSCOPE_API_KEY=your_key
```

### 项目数据
- 将 **课程CSV**（如 `COMPLS_courses.csv` 或你自己的清洗文件）放在项目根目录或 `data/`。
- *(可选)* 准备 `course_reviews.jsonl` 启用口碑RAG（每行一个JSON对象）：
```json
{"code":"COMP9414","term":"24T1","rating":4.2,"difficulty":3.2,"workload":"medium",
 "pros":["评分公平","讲义清晰"],"cons":["deadline紧"],
 "comment":"总体不错，项目含金量高。","author":"匿名","source":"群内调研"}
```

---

## 🧑‍💻 运行方式

### CLI
```bash
python UNSW_Course_Agent.py
```

### Web UI（Gradio）
```bash
python ui_gradio.py
```
打开浏览器访问：`http://127.0.0.1:7860/`

**示例指令**：
- `给我AI两年选课建议 我要project 233 233 不要9414`
- `请给解释`
- `9414 vs 9814 哪个更推荐`
- `COMP9414 最近评价怎么样`
- `导出计划` / `导出日历`

**导出**：同一会话中先生成规划，再点击 **导出CSV/ICS** 下载。

---

## 🔌 程序化接口（Python）
```python
from UNSW_Course_Agent import agent_respond

print(agent_respond("9414是T几"))
print(agent_respond("给我AI两年选课建议 我要project 233 233 不要9414"))
print(agent_respond("请给解释"))
print(agent_respond("COMP9414 评价怎么样"))
```
返回值为**最终文本**（单轮调用）。内部通过 LangGraph 路由并汇总输出。

---

## 🔐 配置项（.env）
- `DASHSCOPE_API_KEY` —— 必填，用于 Qwen/DashScope 推理与嵌入。  
> 推荐仅在本机 `.env` 持有密钥；不要提交到仓库。

---

## 🧱 技术栈
- **LangGraph**：状态图/路由
- **LangChain Community**：DashScope 向量化、FAISS
- **Qwen (DashScope)**：意图/对话
- **FAISS**：可选的语义检索
- **pandas**：CSV 处理
- **Gradio**：Web UI
- **python-dotenv**：环境变量加载

依赖列表见 `requirements.txt`。

---

## 🗂️ 仓库结构
```
.
├─ UNSW_Course_Agent.py          # Backend：LangGraph + agent_respond()
├─ ui_gradio.py                  # Gradio UI（聊天 + 导出按钮）
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .gitattributes
├─ .env.example
├─ course_reviews.jsonl          # (可选) 本地口碑语料
└─ data/                         # 私有CSV可放这里
```

---

## 🧭 使用指引（关键流程）
1) **课程查询**：输入课程号，询问学期/先修/类别/简介（如 `9414是T几`、`COMP9414 的前置`）。  
2) **两年规划**：可指定  
   - 路线偏好：`project/项目/9900` 或 `research/研究/9991`  
   - 学期负荷：`233 233` / `332 332` / `333333`  
   - 排除项：课号或主题（如“不要9414/不要CV”）  
   - 已修清单：如 `9021 9331`  
   生成后输入 `请给解释` 获取逐课理由与数据引用。  
3) **口碑RAG**：`COMP9414 评价怎么样`，或比较 `9414 vs 9814 哪个更推荐`。  
4) **导出**：`导出计划` → `plan.csv`; `导出日历` → `plan.ics`。

---

## 🧩 数据契约
**`course_reviews.jsonl`** 字段：
- `code` *(str)*：如 `COMP9414`
- `term` *(str)*：如 `24T1`
- `rating` *(float 0~5)*
- `difficulty` *(float 0~5)*
- `workload` *("light"|"medium"|"heavy")*
- `pros`/`cons` *(list[str])*
- `comment` *(str)*
- `author` *(str, 可选)*
- `source` *(str, 可选)*

---

## 🛠️ 工程风格 / 规范
- **代码风格**：PEP 8 + 类型注解（能加尽量加）；函数职责单一。
- **提交规范**：Conventional Commits（`feat:`, `fix:`, `docs:` ...）。
- **数据与密钥**：CSV 可本地保留，`.env` 不要提交；忽略向量库与导出文件。
- **可扩展性**：新增意图/流程 → 新建 LangGraph 节点 + 路由正则；新增数据源 → 以“薄适配层”接入。

---

## 🐞 故障排查
- **未配置API Key**：设置 `DASHSCOPE_API_KEY` 后重启。
- **课程CSV为空或缺列**：确保包含 `CourseCode/OfferingTerms/Description` 等核心列。
- **向量库报错**：未构建也可降级运行；不影响基本问答与规划。
- **UI导出不可用**：需先生成规划，再点击导出按钮。

---

## 📄 许可证
MIT — 见 `LICENSE`。

---

## 🙌 致谢
UNSW公开课程信息、LangGraph & Gradio 社区、Qwen/DashScope 团队。


---

# UNSW Course Agent (English)

A **LangGraph**-based chatbot for **UNSW course Q&A and 2‑year AI track planning**, featuring **local-only reviews RAG** and **CSV/ICS exports**. Ships with both **CLI** and **Gradio Web UI**.

> Ready for GitHub submission: clear structure, local runnable, API & engineering guidelines included.

---

## ✨ Features
- **Course facts**: terms (T1/T2/T3), prerequisites/exclusions, category, description.
- **Two‑year AI study plan** (6 terms): respects category quotas, prerequisites, per-term load, and project/research preference.
- **On‑demand explanations**: after the plan, type “**请给解释**” to get per‑course rationale with data references.
- **Local Reviews RAG**: aggregates rating/difficulty/workload and top pros/cons from `course_reviews.jsonl` with short quotes.
- **Exports**:  
  - `plan.csv` — `Term, CourseCode, CourseName, Description`  
  - `plan.ics` — all‑day calendar events (Google/Apple/Outlook).
- **Two entrypoints**: CLI and Gradio UI.

---

## 🚀 Quickstart
```bash
# 1) Virtual env
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Environment (do NOT commit secrets)
cp .env.example .env
# Set DASHSCOPE_API_KEY=your_key in .env
```

### Project Data
- Put your **course CSV** (e.g., `COMPLS_courses.csv`) in the repo root or `data/`.
- *(Optional)* `course_reviews.jsonl` enables Reviews RAG (one JSON per line):
```json
{"code":"COMP9414","term":"24T1","rating":4.2,"difficulty":3.2,"workload":"medium",
 "pros":["Fair grading","Clear slides"],"cons":["Tight deadlines"],
 "comment":"Solid overall; meaningful project.","author":"Anon","source":"Survey"}
```

---

## 🧑‍💻 Run

### CLI
```bash
python UNSW_Course_Agent.py
```

### Web UI (Gradio)
```bash
python ui_gradio.py
```
Then open `http://127.0.0.1:7860/`.

**Example prompts**:
- `给我AI两年选课建议 我要project 233 233 不要9414`
- `请给解释`
- `9414 vs 9814 哪个更推荐`
- `COMP9414 最近评价怎么样`
- `导出计划` / `导出日历`

**Exports**: in the same session, generate a plan first, then click **Export CSV/ICS**.

---

## 🔌 Programmatic API (Python)
```python
from UNSW_Course_Agent import agent_respond

print(agent_respond("9414是T几"))
print(agent_respond("给我AI两年选课建议 我要project 233 233 不要9414"))
print(agent_respond("请给解释"))
print(agent_respond("COMP9414 评价怎么样"))
```
Returns a human‑readable **string** per call. Internally, a LangGraph graph handles routing and aggregation.

---

## 🔐 Configuration (.env)
- `DASHSCOPE_API_KEY` — required for Qwen/DashScope (LLM + embeddings).  
> Keep secrets only in `.env`. Do not commit.

---

## 🧱 Tech Stack
- **LangGraph** (routing, state graph)
- **LangChain Community** (DashScope embeddings, FAISS)
- **Qwen (DashScope)** for intent & chitchat
- **FAISS** for optional semantic retrieval
- **pandas** for CSV
- **Gradio** for Web UI
- **python-dotenv** for env config

Dependencies listed in `requirements.txt`.

---

## 🗂️ Repository Layout
```
.
├─ UNSW_Course_Agent.py          # Backend: LangGraph + agent_respond()
├─ ui_gradio.py                  # Gradio UI (chat + export buttons)
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .gitattributes
├─ .env.example
├─ course_reviews.jsonl          # (optional) local reviews corpus
└─ data/                         # put private CSVs here
```

---

## 🧭 Usage Guide (Key Flows)
1) **Course facts** — ask for terms/prereqs/category/description by code (`9414是T几`, `COMP9414 的前置`).  
2) **Two‑year plan** — specify:  
   - Preference: `project/9900` or `research/9991`  
   - Load pattern: `233 233` / `332 332` / `333333`  
   - Exclusions: codes or topics (`不要9414 / 不要CV`)  
   - Completed: list codes (e.g., `9021 9331`)  
   Then send `请给解释` for per‑course rationale.  
3) **Reviews RAG** — `COMP9414 评价怎么样`, or compare `9414 vs 9814 哪个更推荐`.  
4) **Exports** — `导出计划` → `plan.csv`; `导出日历` → `plan.ics`.

---

## 🧩 Data Contract
**`course_reviews.jsonl`** fields:
- `code` *(str)*, `term` *(str)*, `rating` *(float)*, `difficulty` *(float)*,  
  `workload` *("light"|"medium"|"heavy")*, `pros`/`cons` *(list[str])*,  
  `comment` *(str)*, `author` *(str, optional)*, `source` *(str, optional)*.

---

## 🛠️ Engineering Guidelines
- **Code style**: PEP 8 + type hints; single‑purpose functions.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:` …).
- **Data & secrets**: keep CSVs local; never commit `.env`. Ignore vector DBs and exports.
- **Extensibility**: new intent/flow → add a LangGraph node + router regex; new data source → thin adapter.

---

## 🐞 Troubleshooting
- **Missing API key**: set `DASHSCOPE_API_KEY` and restart.
- **Empty or malformed CSV**: ensure columns like `CourseCode/OfferingTerms/Description` exist.
- **Vector store errors**: features degrade gracefully; core Q&A/plan still work.
- **UI export buttons not working**: generate a plan first, then export.

---

## 📄 License
MIT — see `LICENSE`.

---

## 🙌 Acknowledgements
UNSW public course facts, the LangGraph & Gradio communities, and Qwen/DashScope.
