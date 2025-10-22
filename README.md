# UNSW Course Agent
#### Author by GavinSun

A LangGraph-based chatbot for **UNSW course planning & Q&A**.  
- **CSV facts**: offerings (T1/T2/T3), prerequisites, descriptions
- **2‑year AI study plan** with *on‑demand* explanations (`请给解释`)
- **Reviews RAG** (local JSONL only): recent‑year filter + source weighting
- **Exports**: `plan.csv` and `plan.ics`
- **Web UI**: Gradio; also supports CLI

## Quickstart
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
# then edit .env to set DASHSCOPE_API_KEY
```

### Data
- Put your course CSV (e.g. `COMPLS_courses.csv`) in the **project root** (same folder as the backend script).  
  If you cannot publish the CSV, keep it local and update the path in code/README.
- (Optional) `course_reviews.jsonl` as local reviews corpus; one JSON object per line:
```json
{"code":"COMP9414","term":"24T1","rating":4.2,"difficulty":3.2,"workload":"medium",
  "pros":["评分公平","讲义清晰"],"cons":["deadline紧"],
  "comment":"总体不错，项目含金量高。","author":"匿名","source":"群内调研"}
```

## Run (CLI)
```bash
python UNSW_Course_Agent.py
```

## Run (Web UI, Gradio)
```bash
python ui_gradio.py
```
Then open: <http://127.0.0.1:7860/>

## Export
In the same session **after** generating a plan:
- `导出计划` → creates `plan.csv` (Term, CourseCode, CourseName, Description)
- `导出日历` → creates `plan.ics` (all‑day events; import into Google/Apple/Outlook calendar)

## Repo structure (suggested)
```
.
├─ UNSW_Course_Agent.py          # Backend with LangGraph + agent_respond()
├─ ui_gradio.py                  # Minimal Gradio UI
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .gitattributes
├─ .env.example
├─ course_reviews.jsonl          # (optional) local reviews corpus
└─ data/                         # keep CSV here if not publishing
   └─ README.md
```

## Notes
- **Secrets**: never commit `.env`; keep keys locally.
- **Large/derived files**: vector stores, embeddings, exports are ignored via `.gitignore`.
- **License**: MIT — see `LICENSE`.

---

Made for the LangGraph "Deep Agents" assignment: focus on **explainability**, **tooling (CSV/ICS export)**, and **RAG for reviews only**.
