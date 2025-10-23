# 🧭 UNSW Course Agent  
#### Author: Gavin Sun  

---

## 🇨🇳 中文简介  

**UNSW Course Agent** 是一个基于 **LangGraph** 的课程问答与选课助手，旨在帮助学生高效获取课程信息并制定个性化学习规划。系统结合 **RAG（Retrieval-Augmented Generation）**、**Memory（对话记忆）** 与 **语义检索** 技术，在自然语言交互中实现课程问答、课程推荐、学习计划生成与结果导出。  

### 🔍 核心功能  
- **课程知识问答（Course Q&A）**：支持查询课程编号、开课学期、前置与互斥关系、等价课程及简介。  
- **个性化学习规划（Study Plan Generation）**：基于规则约束与语义分析，自动生成两年学习规划或特定方向的选课方案。  
- **课程推荐（Course Recommendation）**：理解用户输入中的主题（如 *AI*、*Data*、*Security*）、学期与数量，并结合 FAISS 向量检索和课程数据库进行筛选。  
- **课程口碑分析（Review RAG）**：基于本地 JSONL 数据构建的语义索引，支持按时间加权与多源融合的课程评价检索。  
- **结果导出（Export）**：  
  - 生成 `plan.csv`，包含课程代码、名称、学期与简介；  
  - 生成 `plan.ics`，可导入 Google、Apple 或 Outlook 日历。  
- **记忆与上下文理解（Memory）**：支持多轮对话，如“9414 是哪学期开课？”→“它是什么课？”，系统可自动识别“它”的指代关系。  
- **交互方式**：提供 **CLI 模式** 与 **Gradio Web UI**，支持自然语言输入与导出操作。  

### ⚙️ 技术栈  
- **LangGraph**：节点式对话流程控制与状态管理。  
- **DashScope/Qwen**：语义理解与生成模型。  
- **FAISS + Embedding**：课程语义检索与相似度匹配。  
- **pandas + ics**：结果导出（CSV / 日历文件）。  
- **RAG + Memory**：检索增强生成与上下文记忆。  

### 📦 快速启动  
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # 设置 DASHSCOPE_API_KEY
```

### 📂 数据  
- `COMPLS_courses.csv`：基于 **UNSW Handbook** 爬取整理，包含课程结构化信息。  
- `course_reviews.jsonl`：模拟学生口碑数据，用于课程评价检索与推荐优化。  

### 🚀 运行方式  
```bash
# 命令行模式
python UNSW_Course_Agent.py

# Web 界面模式（Gradio）
python ui_gradio.py
# 打开 http://127.0.0.1:7860/
```

### 📤 导出说明  
- **导出计划**：生成 `plan.csv`（课程表及描述）  
- **导出日历**：生成 `plan.ics`（可导入日历系统）  

### 🧩 项目结构  
```
.
├─ UNSW_Course_Agent.py          # LangGraph 主逻辑与对话代理
├─ ui_gradio.py                  # Web UI
├─ course_reviews.jsonl          # 本地口碑数据 (可选)
├─ requirements.txt
├─ .env.example
└─ data/
   └─ COMPLS_courses.csv
```

### 🧠 工程特性  
- 模块化架构，数据层与逻辑层完全解耦。  
- 多级兜底策略：模型异常、数据缺失与输入错误均可安全恢复。  
- 支持单元测试与模块化调试。  
- 生成计划与口碑分析可解释、可追溯。  

### 📜 License  
MIT License  

---

## 🇬🇧 English Introduction  

**UNSW Course Agent** is a **LangGraph-based intelligent assistant** for UNSW course Q&A and study planning.  
It integrates **RAG (Retrieval-Augmented Generation)**, **semantic retrieval**, and **Memory-based dialogue management** to provide natural-language interactions for course search, recommendation, and two-year study planning.  

### 🔍 Key Features  
- **Course Q&A** — Retrieve course details (offering term, prerequisites, exclusions, equivalents, description).  
- **Study Plan Generation** — Automatically build a two-year plan with rule-based and semantic reasoning.  
- **Course Recommendation** — Understands topics (AI, Data, Security), term, and number of courses, then retrieves related subjects via FAISS.  
- **Review RAG** — Local JSONL-based review retrieval with time weighting and source aggregation.  
- **Export** —  
  - `plan.csv`: structured table (Term, Code, Name, Description).  
  - `plan.ics`: calendar events importable to Google / Apple / Outlook.  
- **Memory** — Supports contextual follow-ups like:  
  “When is 9414 offered?” → “What is it about?”  
- **Interface** — CLI and Gradio-based Web UI.  

### ⚙️ Tech Stack  
- **LangGraph** — Dialogue flow orchestration.  
- **DashScope/Qwen** — LLM for intent recognition & NLG.  
- **FAISS + Embedding** — Semantic retrieval.  
- **pandas + ics** — Data export.  
- **RAG + Memory** — Contextual retrieval & multi-turn continuity.  

### 🏃 Quickstart  
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # set DASHSCOPE_API_KEY
```

### 📂 Data  
- `COMPLS_courses.csv`: structured course data (from UNSW Handbook).  
- `course_reviews.jsonl`: synthetic course reviews for RAG retrieval.  

### 💻 Run  
```bash
python UNSW_Course_Agent.py         # CLI
python ui_gradio.py                 # Gradio Web UI
# Open http://127.0.0.1:7860/
```

### 📤 Export  
- `导出计划` → generates `plan.csv`  
- `导出日历` → generates `plan.ics`  

### 📁 Structure  
```
.
├─ UNSW_Course_Agent.py   # Core agent with LangGraph & RAG
├─ ui_gradio.py           # Web interface
├─ course_reviews.jsonl   # Local review dataset
├─ requirements.txt
├─ .env.example
└─ data/COMPLS_courses.csv
```

### ⚡ Notes  
- Modular, decoupled design; safe fallback strategies for errors and missing data.  
- Supports unit tests & modular debugging.  
- Exported plans and explanations are interpretable and reproducible.  
- MIT License.  
