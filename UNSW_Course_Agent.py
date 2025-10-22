from __future__ import annotations

# === Export Helpers: export plan to ICS (calendar) ===
def export_plan_ics(schedule: dict[int, list[str]], terms: list[str], df_all: pd.DataFrame,
                    out_path: str = "plan.ics") -> str:
    import datetime as _dt
    def _term_start_date(term: str, index1: int) -> _dt.date:
        t = (term or "").upper()
        if "T1" in t:  month, day = 3, 1
        elif "T2" in t: month, day = 6, 1
        elif "T3" in t: month, day = 9, 1
        elif "SUMMER" in t or "T0" in t: month, day = 12, 1
        else: month, day = 1, 1
        y = _dt.date.today().year + (0 if index1 <= 3 else 1)
        return _dt.date(y, month, day)
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//UNSW Course Agent//EN"]
    for i, term in enumerate(terms, 1):
        start = _term_start_date(term, i)
        for code in (schedule.get(i, []) or []):
            hit = df_all[df_all["CourseCode"].str.upper()==str(code).upper()]
            name = hit.iloc[0].get("CourseName", "") if not hit.empty else ""
            desc = hit.iloc[0].get("Description", "") if not hit.empty else ""
            desc = (desc or "").replace("\r", " ").replace("\n", " ")
            uid = f"{code}-{term}-{i}@unsw-agent"
            dtstart = start.strftime("%Y%m%d")
            dtend = (start + _dt.timedelta(days=1)).strftime("%Y%m%d")
            lines += [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"SUMMARY:{code} – {name} [{term}]",
                f"DTSTART;VALUE=DATE:{dtstart}",
                f"DTEND;VALUE=DATE:{dtend}",
                f"DESCRIPTION:{desc[:800]}",
                "END:VEVENT",
            ]
    lines += ["END:VCALENDAR"]
    content = "\r\n".join(lines) + "\r\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path
# -*- coding: utf-8 -*-
"""
🎓 UNSW Course Agent
- 多课程严格匹配（DataFrame）+ 尾号/模糊匹配 + 向量检索兜底
- 对话回指：支持“它/这个/前置/互斥/描述”等；Web 端把上下文拼接给后端
- 课程信息查询：开课学期/先修/互斥/等价/类别/简介 全覆盖
- AI 方向两年选课建议（精准毕业规则）
  • 总计 96 UOC；Capstone/Research 18 UOC；非 Capstone 78 UOC（13×6UOC）
  • 严格类别映射 + Elective 白名单 + DKE 列表
  • 路线偏好：Project（COMP9900 + GSOE9010/9011 + ≥1 DKE）
             或 Research（COMP9991+9993；或 9991+9992 + ≥1 DKE）
  • 支持“已修课程”（如 COMP9021/9024/…）、“排除课程/主题”（如 不要9414/不要CV）、
    “学期负载”（233 233 / 332 332 等）
  • 排课策略：先 Foundational Core，再交错 Adv/AI/DKE/Elective；遵守开课学期与先修
  • 解释按需：先生成计划，再说“请给解释”→ 逐门解释 + 引用（来自本地 CSV）
- 课程口碑（RAG，仅本地 JSONL）
  • 汇总 评分/难度/工作量 + 亮点/痛点 + 代表性评论
  • 支持对比：如“9414 vs 9814 哪个更推荐”
- 导出功能
  • “导出计划” → 生成 plan.csv（Term, CourseCode, CourseName, Description）
  • “导出日历” → 生成 plan.ics（每门课 1 条 All-day 事件，可导入 Google/Apple/Outlook）
- Web UI（Gradio）
  • ui_gradio.py 调用 agent_respond()，内置示例问题与一键导出按钮

依赖：
  pip install -U langgraph langchain-community dashscope python-dotenv faiss-cpu pandas gradio

准备：
  - 将 COMPLS_courses.csv 与本文件放同一目录（或在代码中调整路径）
  - 在 .env 中设置 DASHSCOPE_API_KEY
  - （可选）准备 course_reviews.jsonl（每行一个 JSON 对象）

运行：
  # 命令行
  python UNSW_Course_Agent.py

  # 网页版（Gradio）
  python ui_gradio.py   → 打开 http://127.0.0.1:7860/

常用示例：
  - 给我AI两年选课建议 我要project 233 233 不要9414
  - 请给解释
  - 9414在T几 / 它的课程描述 / 它的前置
  - COMP9414 评价怎么样 / 9414 vs 9814 哪个更推荐
  - 导出计划 / 导出日历
"""

import os, re, json
import pandas as pd
from typing import TypedDict, Annotated, List, Any, Optional
from dotenv import load_dotenv

from dashscope import Generation
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ---------------- Config ----------------
load_dotenv()
CSV_CANDIDATES = [
    "COMPLS_courses.csv",
]
CSV_FILE = next((p for p in CSV_CANDIDATES if os.path.exists(p)), CSV_CANDIDATES[-1])
VECTOR_DB_PATH = "course_vector_db_v4"
EMBED_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-turbo"
TOP_K = 6

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ DASHSCOPE_API_KEY 未配置，请在 .env 中设置。")
#print("🔑 DASHSCOPE_API_KEY:", api_key[:8] + "..." + api_key[-4:])

# ---------------- Load Data ----------------
print("📘 加载课程数据:", CSV_FILE)
df = pd.read_csv(CSV_FILE, dtype=str).fillna("")
df["CourseCode"] = df["CourseCode"].astype(str).str.upper().str.replace(" ", "", regex=False)
df = df[df["CourseCode"].str.match(r"^[A-Z]{4}\d{4}$", na=False)].copy()
if df.empty:
    raise RuntimeError("课程数据为空，请检查 CSV。")

# Build retrieval docs (optional)
vectorstore = None
try:
    docs = []
    for _, r in df.iterrows():
        parts = [
            f"{r['CourseCode']} - {r.get('CourseName','')}",
            f"{r.get('Credits','')}UOC, {r.get('OfferingTerms','')}",
            r.get('Description',''),
        ]
        if r.get('Category'): parts.append(f"Category: {r['Category']}")
        if r.get('ConditionsForEnrolment'): parts.append(f"Prereqs: {r['ConditionsForEnrolment']}")
        if r.get('EquivalentCourses'): parts.append(f"Equivalent: {r['EquivalentCourses']}")
        if r.get('ExclusionCourses'): parts.append(f"Exclusion: {r['ExclusionCourses']}")
        docs.append(Document(page_content="\n".join([x for x in parts if str(x).strip()])))
    embeddings = DashScopeEmbeddings(model=EMBED_MODEL, dashscope_api_key=api_key)
    if not os.path.exists(VECTOR_DB_PATH):
        print("🧠 生成向量数据库...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
    else:
        print("📂 加载已有向量数据库...")
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print("⚠️ 检索向量库不可用：", e)
    vectorstore = None



# === Course Reviews Store (RAG 仅用于“课程评价”) ===
import json
from collections import Counter, defaultdict

REVIEWS_FILE = "course_reviews.jsonl"
try:
    _reviews_raw = []
    with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            _reviews_raw.append(json.loads(line))
except FileNotFoundError:
    _reviews_raw = []

# 按课程号聚合
REVIEWS_BY_CODE: dict[str, list[dict]] = defaultdict(list)
for r in _reviews_raw:
    code = str(r.get("code","")).upper().strip()
    if code:
        REVIEWS_BY_CODE[code].append(r)

# === Reviews 配置：近两年过滤 + 来源加权 ===
from datetime import date
import re as _re

def _two_digit_year(y: int) -> str:
    return f"{y%100:02d}"

RECENT_YEARS_DEFAULT: set[str] = {_two_digit_year(date.today().year), _two_digit_year(date.today().year - 1)}

SOURCE_WEIGHTS_DEFAULT: dict[str, float] = {
    "内部问卷": 1.0,
    "群内调研": 0.9,
    "匿名收集": 0.8,
}

def _term_year_prefix(term: str) -> str:
    m = _re.match(r"(\d{2})", str(term or ""))
    return m.group(1) if m else ""

def _is_recent_term(term: str, recent_years: set[str] | None) -> bool:
    if not recent_years:
        return True
    return _term_year_prefix(term) in recent_years

def _src_weight(src: str, weights: dict[str, float] | None) -> float:
    if not weights:
        return 1.0
    return float(weights.get(str(src or ""), 1.0))

def _fmt_pct(x: float) -> str:
    try:
        return f"{x*100:.0f}%"
    except Exception:
        return "0%"

def summarize_reviews_for(code: str) -> str:
    code = str(code).upper().strip()
    items = REVIEWS_BY_CODE.get(code, [])
    if not items:
        return f"暂无 {code} 的评价。可创建 {REVIEWS_FILE} 并追加该课程的评价记录（JSONL）。"

    # 统计
    ratings = [r.get("rating") for r in items if isinstance(r.get("rating"), (int, float))]
    diffs = [r.get("difficulty") for r in items if isinstance(r.get("difficulty"), (int, float))]
    wls = [str(r.get("workload","")).lower() for r in items if r.get("workload")]
    pros = [p for r in items for p in (r.get("pros") or [])]
    cons = [c for r in items for c in (r.get("cons") or [])]

    import statistics as S
    avg_rating = round(S.mean(ratings), 1) if ratings else None
    avg_diff = round(S.mean(diffs), 1) if diffs else None
    wl_cnt = Counter(wls)
    wl_total = sum(wl_cnt.values()) or 1

    def _brief(r):
        t = (r.get("comment") or "").strip().replace("\n"," ")
        if len(t) > 120: t = t[:120] + "..."
        meta = " / ".join([s for s in [r.get("term"), r.get("author"), r.get("source")] if s])
        return f"“{t}” —— {meta}" if t else ""

    reps = [_brief(r) for r in items[:3]]
    reps = [x for x in reps if x]

    lines = [f"⭐ {code} 课程口碑（{len(items)} 条）"]
    if avg_rating is not None: lines.append(f"- 综合评分：{avg_rating}/5")
    if avg_diff is not None: lines.append(f"- 难度：{avg_diff}/5")
    if wl_cnt:
        lines.append(f"- 工作量：heavy {_fmt_pct(wl_cnt.get('heavy',0)/wl_total)} / "
                     f"medium {_fmt_pct(wl_cnt.get('medium',0)/wl_total)} / "
                     f"light {_fmt_pct(wl_cnt.get('light',0)/wl_total)}")
    if pros: 
        from collections import Counter as _C
        top = [w for w,_ in _C(pros).most_common(3)]
        lines.append(f"- 亮点：{'、'.join(top)}")
    if cons:
        from collections import Counter as _C
        top = [w for w,_ in _C(cons).most_common(3)]
        lines.append(f"- 痛点：{'、'.join(top)}")
    if reps:
        lines.append("- 代表性评论：")
        for s in reps:
            lines.append("  · " + s)

    lines.append(f"📎 引用：来自本地评价库 {REVIEWS_FILE}。")
    return "\n".join(lines)

def compare_reviews(a: str, b: str) -> str:
    sa, sb = summarize_reviews_for(a), summarize_reviews_for(b)
    return f"{sa}\n\n——— 对比 ——\n\n{sb}"
# =========================================================
# Last plan cache (for on-demand explain)
# =========================================================
LAST_PLAN_SCHEDULE = None
LAST_PLAN_TERMS = None
LAST_PLAN_STATE = {}

# === Export Helpers: export plan to CSV ===
def export_plan_csv(schedule: dict[int, list[str]], terms: list[str], df_all: pd.DataFrame,
                    out_path: str = "plan.csv", include_desc: bool = True) -> str:
    rows = []
    for i, term in enumerate(terms, 1):
        codes = schedule.get(i, []) or []
        for code in codes:
            hit = df_all[df_all["CourseCode"].str.upper()==str(code).upper()]
            if not hit.empty:
                r = hit.iloc[0]
                rows.append({
                    "Term": term,
                    "CourseCode": str(code).upper(),
                    "CourseName": r.get("CourseName",""),
                    "Description": r.get("Description","") if include_desc else "",
                })
            else:
                rows.append({
                    "Term": term,
                    "CourseCode": str(code).upper(),
                    "CourseName": "",
                    "Description": "",
                })
    try:
        import pandas as _pd
        _pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        import csv
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Term","CourseCode","CourseName","Description"])
            w.writeheader()
            w.writerows(rows)
    return out_path
# =========================================================
# Utilities
# =========================================================
def extract_text(result):
    try:
        if isinstance(result, dict):
            return result.get("output", {}).get("text", "") or result.get("text", "") or str(result)
        if hasattr(result, "output") and isinstance(result.output, dict):
            return result.output.get("text", "")
        if hasattr(result, "output_text"):
            return getattr(result, "output_text", "")
        return str(result)
    except Exception:
        return ""

def safe_answer(content):
    if content is None:
        return {"answer": [HumanMessage(content="⚠️ 没有可返回的内容。")]}
    if isinstance(content, str):
        return {"answer": [HumanMessage(content=content)]}
    if isinstance(content, list):
        msgs = []
        for c in content:
            msgs.append(HumanMessage(content=str(c)) if not isinstance(c, HumanMessage) else c)
        return {"answer": msgs}
    return {"answer": [HumanMessage(content=str(content))]}

def _extract_latest_user_utterance(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    patterns = [
        r"(?:^|\n)\s*用户[:：](.+?)(?=\n\s*助手[:：]|$)",
        r"(?:^|\n)\s*User[:：](.+?)(?=\n\s*(?:Assistant|Bot)[:：]|$)",
        r"(?:^|\n)\s*You[:：](.+?)(?=\n\s*(?:Assistant|Bot)[:：]|$)",
    ]
    matches = []
    for pat in patterns:
        matches += list(re.finditer(pat, text, flags=re.S | re.I))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()

def _recent_codes_from_context(raw_text: str, limit: int = 1):
    text = str(raw_text or "").upper()
    codes = re.findall(r"(?<![A-Z0-9])([A-Z]{4}\d{4})(?![A-Z0-9])", text)
    if not codes:
        return []
    picked, seen = [], set()
    for c in reversed(codes):
        if c not in seen:
            picked.append(c); seen.add(c)
            if len(picked) >= limit:
                break
    return list(reversed(picked))

# =========================================================
# Parsing & lookups
# =========================================================
COURSE_PREFIX_TRY = ["COMP","ENGG","INFS","MATH","SENG","DATA"]

def _extract_candidates_from_query(q: str):
    qU = str(q or "").upper()
    full = re.findall(r"(?<![A-Z0-9])([A-Z]{4}\d{4})(?![A-Z0-9])", qU)
    nums = re.findall(r"(?<![A-Z0-9])(\d{4})(?![A-Z0-9])", qU)
    cands = list(dict.fromkeys(full))
    for n in nums:
        for pre in COURSE_PREFIX_TRY:
            c = f"{pre}{n}"
            if c not in cands:
                cands.append(c)
    return cands, nums

def lookup_strict(query: str) -> Optional[pd.DataFrame]:
    q = str(query or "")
    cands, nums = _extract_candidates_from_query(q)
    hits = pd.DataFrame()
    for code in cands:
        hit = df[df["CourseCode"].str.upper().eq(code)]
        if not hit.empty:
            hits = pd.concat([hits, hit])
    for n in nums:
        tail_hit = df[df["CourseCode"].str.upper().str.endswith(n)]
        if not tail_hit.empty:
            hits = pd.concat([hits, tail_hit])
    if hits.empty:
        return None
    hits = hits.drop_duplicates(subset="CourseCode").sort_values("CourseCode").reset_index(drop=True)
    return hits

def lookup_relaxed(query: str) -> Optional[pd.Series]:
    q = str(query or "")
    cands, nums = _extract_candidates_from_query(q)
    for code in cands:
        hit = df[df["CourseCode"].str.upper().eq(code)]
        if not hit.empty:
            return hit.iloc[0]
    for n in nums:
        tail_hit = df[df["CourseCode"].str.upper().str.endswith(n)]
        if not tail_hit.empty:
            comp_first = tail_hit[tail_hit["CourseCode"].str.upper().str.startswith("COMP")]
            return (comp_first.iloc[0] if not comp_first.empty else tail_hit.sort_values("CourseCode").iloc[0])
    for code in cands:
        hit = df[df["CourseCode"].str.upper().str.contains(re.escape(code), na=False)]
        if not hit.empty:
            return hit.iloc[0]
    if vectorstore is not None:
        try:
            res = vectorstore.similarity_search(str(query), k=1)
            if res:
                header = res[0].page_content.split("\n", 1)[0]
                m = re.search(r"(?<![A-Z0-9])([A-Z]{4}\d{4})(?![A-Z0-9])", header)
                if m:
                    code2 = m.group(1)
                    hit = df[df["CourseCode"].str.upper().eq(code2)]
                    if not hit.empty:
                        return hit.iloc[0]
        except Exception:
            pass
    return None

# =========================================================
# Degree rules (AI) — strict categories + elective whitelist + capstone rules
# =========================================================
# —— 目标学分：非Capstone/Research总计 78 UOC（13 × 6UOC）
NON_CAPSTONE_TARGET_UOC = 78
CAPSTONE_TARGET_UOC = 18

# —— 个别课程的学分覆盖（若CSV里无准确UOC，这里兜底）
UOC_OVERRIDE = {
    "COMP9991": 6,   # Research A
    "COMP9992": 6,   # Research B
    "COMP9993": 12,  # Research C
}

def _uoc_of(row) -> int:
    code = str(row.get("CourseCode","")).upper()
    if code in UOC_OVERRIDE:
        return int(UOC_OVERRIDE[code])
    try:
        return int(str(row.get("Credits","") or "0").strip())
    except:
        m = re.search(r"(\d+)", str(row.get("Credits","")))
        return int(m.group(1)) if m else 0

# —— 精确类别映射
AI_RULES = {
    "name": "AI",
    "total_uoc": 96,
    "slots_per_term": 3,   # 每学期 3 门
    "years": 2,            # 两年
    "buckets": [
        {"key": "found_core", "title": "基础核心课",     "tags": ["foundational core courses"],                 "uoc_min": 18, "uoc_max": 18},
        {"key": "adv_core",   "title": "高级核心课",     "tags": ["advanced core courses"],                    "uoc_min": 18, "uoc_max": 18},
        {"key": "ai_core",    "title": "AI核心课",       "tags": ["artificial intelligence core courses","ai core"], "uoc_min": 6,  "uoc_max": 6},
        {"key": "dke",        "title": "学科选修（DKE）","tags": ["disciplinary knowledge elective courses","dke"],   "uoc_min": 18, "uoc_max": 18},
        {"key": "elective",   "title": "一般选修",       "tags": ["electives"],                                "uoc_min": 18, "uoc_max": 18},
        {"key": "project",    "title": "毕业研究/项目",  "tags": ["research","capstone","project","research/capstone/project","capston"], "uoc_min": 18, "uoc_max": 18},
    ]
}

# —— 自动分类：当 CSV Category 为空时，用课程号兜底归类
FOUND_CORE = {"COMP9020","COMP9021","COMP9024"}
ADV_CORE   = {"COMP9311","COMP9331"}
AI_CORE    = {"COMP9414","COMP9814"}
PROJECT_ROUTE = {"COMP9900","GSOE9010","GSOE9011"}
RESEARCH_ROUTE= {"COMP9991","COMP9992","COMP9993"}
# 常见 AI 方向 DKE
DKE_CODES  = {"COMP4418","COMP9417","COMP9418","COMP9434","COMP9444","COMP9491","COMP9517","COMP9727"}

# Elective（一般选修）
ELECTIVE_CODES = {
"COMP4121","COMP4128","COMP4141","COMP4161","COMP4418","COMP4511",
"COMP6080","COMP6131","COMP6441","COMP6443","COMP6445","COMP6447",
"COMP6448","COMP6451","COMP6452","COMP6453","COMP6713","COMP6714",
"COMP6721","COMP6733","COMP6741","COMP6752","COMP6771","COMP6841",
"COMP6843","COMP6845","COMP6991","COMP9032","COMP9044","COMP9101",
"COMP9102","COMP9153","COMP9154","COMP9164","COMP9201","COMP9211",
"COMP9222","COMP9242","COMP9243","COMP9283","COMP9312","COMP9313",
"COMP9315","COMP9319","COMP9321","COMP9332","COMP9333","COMP9334",
"COMP9336","COMP9337","COMP9414","COMP9415","COMP9417","COMP9418",
"COMP9434","COMP9444","COMP9447","COMP9491","COMP9511","COMP9517",
"COMP9727","COMP9814","COMP9920","GSOE9210","GSOE9220",
"MATH5836","MATH5845","MATH5855","MATH5905","MATH5960"
}

def _auto_category_by_code(code: str) -> str:
    u = str(code).upper().strip()
    if u in FOUND_CORE: return "foundational core courses"
    if u in ADV_CORE:   return "advanced core courses"
    if u in AI_CORE:    return "artificial intelligence core courses"
    if u in PROJECT_ROUTE or u in RESEARCH_ROUTE: return "research/capstone/project"
    if u in DKE_CODES:  return "disciplinary knowledge elective courses"
    if u in ELECTIVE_CODES: return "electives"
    return ""   # ⚠️ 不归类，避免把未知课程误判成选修

def _row_auto_cat(row):
    raw = str(row.get("Category","")).strip().lower()
    return raw if raw else _auto_category_by_code(row.get("CourseCode",""))

df["AutoCategory"] = df.apply(_row_auto_cat, axis=1)

# —— helpers

def _parse_completed_codes(s: str) -> set:
    """
    解析用户声明的“已修课程”：
    - 支持完整课号：COMP9021 / MATH5845 / GSOE9011 ...
    - 支持裸四位数字：9021 -> 默认映射为 COMP9021
    - 返回大写去重后的课程号集合
    """
    text = str(s or "").upper()

    # 1) 先取完整课号（如 COMP9021 / MATH5845）
    full = set(re.findall(r"(?<![A-Z0-9])([A-Z]{4}\d{4})(?![A-Z0-9])", text))

    # 2) 再取裸四位数字并默认映射为 COMP####
    nums = re.findall(r"(?<![A-Z0-9])(\d{4})(?![A-Z0-9])", text)
    for n in nums:
        full.add(f"COMP{n}")

    return full


def _parse_term_loads(s: str) -> list | None:
    """
    从用户输入里解析 6 个学期的课程门数（每学期 1~3 门）。
    支持示例：
      - "332 332" / "233 233" / "333333"
      - "3-3-2 3-3-2" / "3,3,2,3,3,2"
      - "233233"（紧凑写法）
    返回 list[int] 长度=6；不合法时返回 None。
    """
    txt = str(s or "").strip()
    if not txt:
        return None

    # 把分隔符统一成空格
    t = re.sub(r"[^0-9]", " ", txt)
    nums = [n for n in t.split() if n.isdigit()]

    # 情况A：直接给了 6 个数字
    if len(nums) >= 6:
        loads = [int(x) for x in nums[:6]]
    else:
        # 情况B：可能给了连写的 6 位，比如 "233233"
        comp = "".join(nums)
        if len(comp) == 6 and comp.isdigit():
            loads = [int(c) for c in comp]
        else:
            return None

    # 校验范围 1~3
    if any(x < 1 or x > 3 for x in loads):
        return None
    return loads


# === Exclusion parsing (course codes / topics) ===
NEG_PAT = r"(不要|别|不想|排除|去掉|换掉|exclude|drop)"
CODE_PAT = r"(?:COMP|MATH|ELEC|GSOE)\s*-?\s*\d{4}"

TOPIC_SYNS = {
    "ai": {"ai", "人工智能"},
    "ml": {"ml", "机器学习"},
    "cv": {"cv", "计算机视觉", "vision"},
    "nlp": {"nlp", "自然语言处理"},
}

def parse_exclusions(q: str):
    q_low, q_up = (q or "").lower(), (q or "").upper()
    exclude_codes, exclude_topics = set(), set()
    if re.search(NEG_PAT, q_low):
        for c in re.findall(CODE_PAT, q_up):
            exclude_codes.add(re.sub(r"[\s-]+","",c))
        for k, vs in TOPIC_SYNS.items():
            if any(v in q_low for v in vs):
                exclude_topics.add(k)
    return exclude_codes, exclude_topics

def build_search_text(df_):
    return (
        df_["CourseName"].fillna("") + " " +
        df_["Description"].fillna("") + " " +
        df_["AutoCategory"].fillna("")
    ).str.lower()

def apply_exclusions(df_, exclude_codes, exclude_topics):
    df = df_.copy()
    if not exclude_codes and not exclude_topics:
        return df
    df["__search__"] = build_search_text(df)
    mask = ~df["CourseCode"].str.upper().isin({c.upper() for c in exclude_codes})
    if exclude_topics:
        topic_kw = "|".join(map(re.escape, exclude_topics))
        mask &= ~df["__search__"].str.contains(topic_kw, case=False, na=False)
    return df[mask].drop(columns=["__search__"])

def _row_from_code_safe(code: str, uniq: dict, df_: pd.DataFrame):
    """
    安全地通过课程号拿到一行记录：
    - 如果在 uniq 字典里有（排课时缓存的行），优先取
    - 否则从 df_ 里按 CourseCode 精确匹配
    """
    c = str(code).upper().strip()
    if isinstance(uniq, dict) and c in uniq:
        return uniq[c]
    hit = df_[df_["CourseCode"].str.upper().eq(c)]
    return hit.iloc[0] if not hit.empty else None


def _norm(s: str) -> str:
    return (str(s or "").strip().lower())

def _category_match(cat: str, tag_list: list) -> bool:
    c = _norm(cat)
    return any(t.lower() in c for t in tag_list if t)

# 放到常量区任意位置，便于复用
CAPSTONE_CODES = set(PROJECT_ROUTE) | set(RESEARCH_ROUTE)

def _offered_in_term(row, term: str) -> bool:
    terms = str(row.get("OfferingTerms","") or "").upper().strip()
    if not terms:
        return True  # 兜底：未知学期视为可排（用于学习计划）
    if "NOT OFFERED" in terms:
        return False
    return term in terms or terms in {"ALL", "ANY"}

def _parse_codes_list(s: str):
    s = (s or "").upper().replace("，",";").replace(",",";")
    parts = [p.strip() for p in re.split(r"[;|/ ]+", s) if p.strip()]
    return [p for p in parts if re.match(r"^[A-Z]{4}\d{4}$", p)]

def _prereq_codes(row) -> set:
    text = str(row.get("ConditionsForEnrolment","") or "")
    low = text.lower()
    # 仅在明确出现“先修/修读要求”时，才识别先修课
    if not re.search(r"prereq|pre-?requisite|assumed knowledge|must have completed|completion of", low):
        return set()
    # 提取课程号，并限制在常见学院前缀
    codes = set(_parse_codes_list(text))
    allowed_prefixes = ("COMP","MATH","GSOE","ENGG","DATA","SENG","INFS")
    return {c for c in codes if c.startswith(allowed_prefixes)}


def _collect_bucket_courses(df_, bucket_cfg):
    # 先按 AutoCategory/Category 匹配（小写包含）
    tags = [t.lower() for t in bucket_cfg["tags"]]
    def _cat_of_row(row):
        return str(row.get("AutoCategory", row.get("Category",""))).lower().strip()

    hit = df_[df_.apply(lambda r: any(t in _cat_of_row(r) for t in tags), axis=1)]

    # 兜底：按“课程号集合”强制补齐（解决 Category 为空的问题）
    if hit.empty:
        key = bucket_cfg["key"]
        if key == "found_core":
            hit = df_[df_["CourseCode"].str.upper().isin(FOUND_CORE)]
        elif key == "adv_core":
            hit = df_[df_["CourseCode"].str.upper().isin(ADV_CORE)]
        elif key == "ai_core":
            hit = df_[df_["CourseCode"].str.upper().isin(AI_CORE)]
        elif key == "dke":
            hit = df_[df_["CourseCode"].str.upper().isin(DKE_CODES)]
        elif key == "elective":
            hit = df_[df_["CourseCode"].str.upper().isin(ELECTIVE_CODES)]
        elif key == "project":
            hit = df_[df_["CourseCode"].str.upper().isin(PROJECT_ROUTE | RESEARCH_ROUTE)]

    # 一般选修里排除带“disciplinary”的，避免吃掉 DKE
    if bucket_cfg["key"] == "elective":
        hit = hit[~hit.apply(lambda r: "disciplinary" in _cat_of_row(r), axis=1)]

    # 兜底：AI 核心再保证一下
    if bucket_cfg["key"] == "ai_core" and hit.empty:
        hit = df_[df_["CourseCode"].str.upper().isin(["COMP9414","COMP9814"])]

    return hit.copy()

def _choose_courses_for_bucket(df_, bucket_cfg, already: set, prefer_topic="AI", exclude_codes=None, exclude_topics=None):
    cand = _collect_bucket_courses(df_, bucket_cfg)
    if exclude_codes is not None or exclude_topics is not None:
        cand = apply_exclusions(cand, exclude_codes or set(), exclude_topics or set())
    if cand.empty:
        return cand
    def score_row(r):
        s = 0
        desc = (str(r.get("Description","")) + " " + str(r.get("CourseName",""))).lower()
        if prefer_topic.lower() in desc: s += 3
        terms = str(r.get("OfferingTerms","") or "").upper()
        s += len([t for t in ["T1","T2","T3"] if t in terms])  # 开课越多越好
        eq = set(_parse_codes_list(r.get("EquivalentCourses","")))
        ex = set(_parse_codes_list(r.get("ExclusionCourses","")))
        if already & (eq | ex): s -= 10
        return s
    cand = cand[~cand["CourseCode"].str.upper().isin(already)].copy()
    if cand.empty:
        return cand
    cand["_score"] = cand.apply(score_row, axis=1)
    cand = cand.sort_values(["_score","CourseCode"], ascending=[False, True])
    return cand

def _select_capstone_combo(df_, prefer: str = "auto"):
    """
    返回 (capstone_rows, route_type)：
    - Project：COMP9900 + (GSOE9010 或 GSOE9011) + 1 门 DKE（18 UOC）
    - Research：
        * 9991 + 9993（9993=12UOC）=> 共18UOC，无需DKE
        * 若无9993，则 9991 + 9992 + 1 门 DKE => 共18UOC
    prefer: 'project' / 'research' / 'auto'
    """
    def row_of(code):
        hit = df_[df_["CourseCode"].str.upper().eq(code)]
        return hit.iloc[0] if not hit.empty else None
    have = lambda c: not df_[df_["CourseCode"].str.upper().eq(c)].empty

    # Research 优先（当 prefer=research 或 auto 且资源可用）
    if prefer in ("research", "auto"):
        if have("COMP9991") and have("COMP9993"):
            r1, r3 = row_of("COMP9991"), row_of("COMP9993")
            return [r1, r3], "Research 路线（COMP9991 + COMP9993，无需额外DKE）"
        if have("COMP9991") and have("COMP9992"):
            r1, r2 = row_of("COMP9991"), row_of("COMP9992")
            dke_pool = df_[df_["CourseCode"].str.upper().isin(DKE_CODES)]
            dke_pool = dke_pool[~dke_pool["CourseCode"].str.upper().isin({"COMP9991","COMP9992"})]
            if dke_pool.empty:
                dke_pool = df_[
                    (df_["AutoCategory"].str.lower().str.contains("disciplinary")) &
                    (~df_["CourseCode"].str.upper().isin({"COMP9991","COMP9992"}))
                ]
            dke_row = dke_pool.iloc[0] if not dke_pool.empty else None
            rows = [r1, r2] + ([dke_row] if dke_row is not None else [])
            return rows, "Research 路线（COMP9991 + COMP9992 + 至少1门DKE）"

    # Project
    if have("COMP9900") and (have("GSOE9010") or have("GSOE9011")):
        p = row_of("COMP9900")
        g = row_of("GSOE9011") if have("GSOE9011") else row_of("GSOE9010")
        dke_pool = df_[df_["CourseCode"].str.upper().isin(DKE_CODES)]
        dke_pool = dke_pool[~dke_pool["CourseCode"].str.upper().isin({"COMP9900","GSOE9010","GSOE9011"})]
        if dke_pool.empty:
            dke_pool = df_[(df_["AutoCategory"].str.lower().str.contains("disciplinary")) &
                           (~df_["CourseCode"].str.upper().isin({"COMP9900","GSOE9010","GSOE9011"}))]
        dke_row = dke_pool.iloc[0] if not dke_pool.empty else None
        rows = [p, g] + ([dke_row] if dke_row is not None else [])
        return rows, "Project 路线（COMP9900 + GSOE9010/9011 + 至少1门DKE）"

    return [], "未确定"

def _schedule_ai_plan(df_, rules=AI_RULES, completed: set = None, prefer_route: str = "auto", term_loads: list | None = None, exclude_codes: set | None = None, exclude_topics: set | None = None):
    """
    弹性学期负载 + 先修3门基础核心 + 其余穿插：
    - 学期负载：用户可指定（每学期1~3门）；否则默认 DEFAULT_LOADS
    - 阶段1：Foundational Core（凑满18UOC=3门），尽量放到最前面的学期
    - 阶段2：Adv Core + AI Core + DKE + Elective 按优先级交错放入（不强制“选修必须最后”）
    - Capstone/Research：先选组合，但优先尝试放在后两学期（若先修/开课不允许，会自动前移）
    - 满足非Capstone 78UOC + 各桶最低学分（Found18 / Adv18 / AI6 / DKE18 / Elective18）
    """
    DEFAULT_LOADS = [3,3,2, 3,3,2]  # 没指定时的常用负载
    loads = term_loads if (isinstance(term_loads, list) and len(term_loads)==6) else DEFAULT_LOADS
    # 防御：限制范围1~3
    loads = [min(3, max(1, int(x))) for x in loads]

    completed = set((completed or set()))
    taken_codes = set(code.upper() for code in completed)

    # 0) 先选 Capstone/Research 组合（不立即排课，只作为候选）
    cap_rows, route_type_hint = _select_capstone_combo(df_, prefer=prefer_route)
    cap_codes = [str(r["CourseCode"]).upper() for r in cap_rows if r is not None]

    # 1) 为非Capstone 5 桶构建候选池（先 Found → 再 Adv/AI/DKE/Elective）
    taken_codes.update(cap_codes)
    chosen_by_bucket = {b["key"]: [] for b in rules["buckets"]}
    uoc_by_bucket = {b["key"]: 0 for b in rules["buckets"]}

    # 阶段1：基础核心（只取到18UOC=3门）
    for key in ["found_core"]:
        bcfg = next(b for b in rules["buckets"] if b["key"] == key)
        need_uoc = bcfg["uoc_min"]
        cand = _choose_courses_for_bucket(df_, bcfg, already=taken_codes, exclude_codes=exclude_codes or set(), exclude_topics=exclude_topics or set())
        if cand is not None and not cand.empty:
            for _, r in cand.iterrows():
                code = str(r["CourseCode"]).upper()
                if code in taken_codes:
                    continue
                chosen_by_bucket[key].append(r)
                uoc_by_bucket[key] += _uoc_of(r)
                taken_codes.add(code)
                if uoc_by_bucket[key] >= need_uoc:
                    break

    # 阶段2：其余桶依次补齐到“最低学分”，从：高级核心 → AI核心 → DKE → Elec 但是只有26年 难受
    phase2_order = ["adv_core","ai_core","dke","elective"]
    for key in phase2_order:
        bcfg = next(b for b in rules["buckets"] if b["key"] == key)
        need_uoc = bcfg["uoc_min"]
        cand = _choose_courses_for_bucket(df_, bcfg, already=taken_codes, exclude_codes=exclude_codes or set(), exclude_topics=exclude_topics or set())
        if cand is None or cand.empty:
            continue
        for _, r in cand.iterrows():
            code = str(r["CourseCode"]).upper()
            if code in taken_codes:
                continue
            chosen_by_bucket[key].append(r)
            uoc_by_bucket[key] += _uoc_of(r)
            taken_codes.add(code)
            if uoc_by_bucket[key] >= need_uoc:
                break

    # 阶段3：如非Capstone < 78UOC，再从 DKE → Elective 继续补，直到 78UOC
    def total_noncap_uoc():
        return sum(_uoc_of(x) for ks in ["found_core","adv_core","ai_core","dke","elective"]
                   for x in chosen_by_bucket[ks])
    for key in ["dke","elective","adv_core","ai_core"]:  # 再给些回旋空间
        if total_noncap_uoc() >= NON_CAPSTONE_TARGET_UOC: break
        bcfg = next(b for b in rules["buckets"] if b["key"] == key)
        cand = _choose_courses_for_bucket(df_, bcfg, already=taken_codes, exclude_codes=exclude_codes or set(), exclude_topics=exclude_topics or set())
        if cand is None or cand.empty:
            continue
        for _, r in cand.iterrows():
            if total_noncap_uoc() >= NON_CAPSTONE_TARGET_UOC: break
            code = str(r["CourseCode"]).upper()
            if code in taken_codes:
                continue
            chosen_by_bucket[key].append(r)
            taken_codes.add(code)

    # 2) 排期：先 Found（三门尽量放最前）
    terms = ["T1","T2","T3","T1","T2","T3"]
    schedule = {i+1: [] for i in range(6)}
    term_cap = loads[:]               # 每学期最多门数
    placed = set(completed)

    def try_place(course_row, term_idx):
        if term_cap[term_idx] <= 0:
            return False
        code = str(course_row["CourseCode"]).upper()
        if code in placed:
            return False
        prereq = _prereq_codes(course_row)
        if not prereq.issubset(placed):
            return False
        if not _offered_in_term(course_row, terms[term_idx]):
            return False
        schedule[term_idx+1].append(code)
        placed.add(code)
        term_cap[term_idx] -= 1
        return True

    # 放置顺序 A：Found（尽量 0→1→2→3→4→5）
    early_order = [0,1,2,3,4,5]
    for r in chosen_by_bucket.get("found_core", []):
        for ti in early_order:
            if try_place(r, ti): break

    # 放置顺序 B：其余桶穿插（优先级：Adv → AI → DKE → Elective），仍以早学期优先
    for key in ["adv_core","ai_core","dke","elective"]:
        for r in chosen_by_bucket.get(key, []):
            for ti in early_order:
                if try_place(r, ti): break

    # 放置顺序 C：Capstone（尽量 5→4→3→2→1→0）
    cap_order = [5,4,3,2,1,0]
    for r in cap_rows:
        if r is None:
            continue
        for ti in cap_order:
            if try_place(r, ti):
                break

    # 3) 缺口统计（同 v4.4 口径）
    uniq = {}
    for i in range(1, 7):
        for c in schedule[i]:
            if c not in uniq:
                hit = df_[df_["CourseCode"].str.upper().eq(c)]
                if not hit.empty:
                    uniq[c] = hit.iloc[0]

    flat = [c for term in schedule.values() for c in term]
    planned_set = set(flat)
    bucket_tags = {b["key"]: b["tags"] for b in rules["buckets"]}
    missing = []

    cap_planned = planned_set & set(cap_codes)
    non_cap_planned = planned_set - cap_planned

    # 各桶下限
    for key in ["found_core","adv_core","ai_core","dke","elective"]:
        want = next(b for b in rules["buckets"] if b["key"]==key)["uoc_min"]
        got = 0
        for c in non_cap_planned:
            row = _row_from_code_safe(c, uniq, df_)
            if row is None:
                continue
            if _category_match(str(row.get("AutoCategory", row.get("Category",""))), bucket_tags[key]):
                got += _uoc_of(row)
        if got < want:
            missing.append((key, want - got))

    # 非Cap 总 UOC
    noncap_total = 0
    for c in non_cap_planned:
        row = _row_from_code_safe(c, uniq, df_)
        if row is not None:
            noncap_total += _uoc_of(row)
    if noncap_total < NON_CAPSTONE_TARGET_UOC:
        missing.append(("noncap_total", NON_CAPSTONE_TARGET_UOC - noncap_total))

    # Capstone UOC（目标 18）
    cap_total = 0
    for c in cap_planned:
        row = _row_from_code_safe(c, uniq, df_)
        if row is not None:
            cap_total += _uoc_of(row)
    if cap_total < CAPSTONE_TARGET_UOC:
        missing.append(("project", CAPSTONE_TARGET_UOC - cap_total))

    # 4) 路线类型（以最终排入的课程为准）
    final_route = "未确定"
    all_codes = set([c.upper() for term in schedule.values() for c in term])
    if {"COMP9900"} & all_codes and ({"GSOE9010"} & all_codes or {"GSOE9011"} & all_codes):
        final_route = "Project 路线（COMP9900 + GSOE9010/9011 + 至少1门DKE）"
    elif {"COMP9991"} & all_codes:
        if {"COMP9993"} & all_codes:
            final_route = "Research 路线（COMP9991 + COMP9993，无需额外DKE）"
        elif {"COMP9992"} & all_codes:
            final_route = "Research 路线（COMP9991 + COMP9992 + 至少1门DKE）"

    return schedule, terms, missing, final_route

# =========================================================
# Intent detection / Chitchat / Nodes
# =========================================================
def detect_intent_node(state):
    latest = _extract_latest_user_utterance(state.get("query", ""))
    prompt = f"""
你是一个智能课程助手，请根据用户输入判断其意图。
输出 JSON：
{{
  "intent": "chitchat / term_query / recommend / plan / search",
  "course_code": "(如果提到课程号)",
  "topic": "(如AI、数据、编程)",
  "num_courses": "(数字或None)"
}}
只输出JSON，不要解释。
用户输入：「{latest}」
""".strip()
    intent = {"intent": "search", "course_code": None, "topic": None, "num_courses": None}
    try:
        result = Generation.call(model=LLM_MODEL, prompt=prompt)
        output = extract_text(result).strip()
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and "output" in parsed:
                inner_text = parsed["output"].get("text", "")
                if inner_text.strip().startswith("{"):
                    parsed = json.loads(inner_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            intent.update(parsed)
    except Exception:
        pass
    return {
        "intent": intent.get("intent", "search"),
        "course_code": intent.get("course_code"),
        "topic": intent.get("topic"),
        "num_courses": intent.get("num_courses"),
        "detail_type": "",
        "route_pref": state.get("route_pref","auto"),
        "next_node": intent.get("intent", "search"),
    }

def chitchat_node(state):
    raw = state["query"]
    query = _extract_latest_user_utterance(raw)
    user_lower = query.lower()
    if any(g in user_lower for g in ["你好","hello","hi","嗨","在吗","thanks","谢谢","早上好","下午好","晚上好"]):
        return safe_answer("😊 我在～想了解哪门课？")
    prompt = f"""
你是一位友好、轻松、语气自然的大学助手。请根据用户的输入，用一句中文自然回复：口语化、亲切、不啰嗦、不换行。允许至多一个表情。
用户说：「{query}」
现在回复：
""".strip()
    try:
        result = Generation.call(model=LLM_MODEL, prompt=prompt)
        text = extract_text(result).strip() or "🤖 我在呢～想了解哪门课？"
    except Exception:
        text = "🤖 我在呢～想了解哪门课？"
    return safe_answer(text)

def search_course_node(state):
    raw_query = state.get("query", "")
    query = _extract_latest_user_utterance(raw_query)
    row = lookup_relaxed(query)
    if row is not None:
        r = row
        info = [f"📘 {r['CourseCode']} - {r.get('CourseName','')} ({r.get('Credits','')}UOC, {r.get('OfferingTerms','')})"]
        if r.get("AutoCategory") or r.get("Category"):
            info.append(f"📂 类别：{r.get('AutoCategory') or r.get('Category')}")
        if r.get("ConditionsForEnrolment"): info.append(f"🔑 前置：{r['ConditionsForEnrolment']}")
        if r.get("EquivalentCourses"): info.append(f"🔁 等价：{r['EquivalentCourses']}")
        if r.get("ExclusionCourses"): info.append(f"🚫 互斥：{r['ExclusionCourses']}")
        if r.get("Description"): info.append(f"📝 {r['Description']}")
        return safe_answer("\n".join(info))
    if vectorstore is not None:
        results = vectorstore.similarity_search(query, k=TOP_K)
        if results:
            answer = [f"{i+1}. 📘 {r.page_content}" for i, r in enumerate(results)]
            return safe_answer(answer)
    return safe_answer("❌ 没找到相关课程。")

def term_query_node(state):
    raw_query = state.get("query", "")
    query = _extract_latest_user_utterance(raw_query)
    rows = lookup_strict(query)
    if rows is None or rows.empty:
        recent = _recent_codes_from_context(raw_query, limit=1)
        if recent:
            rows = df[df["CourseCode"].str.upper().isin([c.upper() for c in recent])]
        if rows is None or rows.empty:
            return safe_answer("❌ 未识别到课程号。")
    outputs = []
    for _, r in rows.iterrows():
        code = str(r.get("CourseCode", "")).strip()
        terms = str(r.get("OfferingTerms", "")).strip() or "N/A"
        if terms.lower() in ["not offered", "nan", "none", "", "n/a"]:
            outputs.append(f"⚠️ {code} 当前未在官方Term列表中开设。")
        else:
            outputs.append(f"📅 {code} 在 {terms} 开课。")
    return safe_answer("\n".join(outputs))

def detail_query_node(state):
    raw_query = state.get("query", "")
    detail = state.get("detail_type", "")
    query = _extract_latest_user_utterance(raw_query)
    rows = lookup_strict(query)
    if rows is None or rows.empty:
        recent = _recent_codes_from_context(raw_query, limit=3)
        if recent:
            rows = df[df["CourseCode"].str.upper().isin([c.upper() for c in recent])]
    if rows is None or rows.empty:
        _, nums = _extract_candidates_from_query(query)
        hint = f"（请尝试完整写法，如 COMP{nums[0]}）" if nums else ""
        return safe_answer(f"❌ 没找到相关课程。{hint}".strip())
    outputs = []
    for _, r in rows.iterrows():
        code = str(r.get("CourseCode", "")).strip()
        if detail == "prereq":
            val = str(r.get("ConditionsForEnrolment", "")).strip() or "（未在Handbook明确给出）"
            outputs.append(f"🔑 {code} 的前置/限修：{val}")
        elif detail == "exclusion":
            val = str(r.get("ExclusionCourses", "")).strip() or "（无）"
            outputs.append(f"🚫 {code} 的互斥课程：{val}")
        elif detail == "equivalent":
            val = str(r.get("EquivalentCourses", "")).strip() or "（无）"
            outputs.append(f"🔁 {code} 的等价课程：{val}")
        elif detail == "category":
            val = str(r.get("AutoCategory", r.get("Category",""))).strip() or "（未标注类别）"
            outputs.append(f"📂 {code} 属于：{val}")
        elif detail == "desc":
            name = str(r.get("CourseName","")).strip()
            desc = str(r.get("Description","")).strip() or "（暂无描述）"
            outputs.append(f"📝 {code}{' - ' + name if name else ''} 的课程描述：{desc}")
        else:
            parts = [
                f"📘 {code} - {str(r.get('CourseName',''))} ({str(r.get('Credits',''))}UOC, {str(r.get('OfferingTerms',''))})",
            ]
            if str(r.get("AutoCategory", r.get("Category",""))).strip():
                parts.append(f"📂 类别：{r.get('AutoCategory', r.get('Category',''))}")
            if str(r.get("ConditionsForEnrolment","")).strip():
                parts.append(f"🔑 前置：{r.get('ConditionsForEnrolment')}")
            if str(r.get("EquivalentCourses","")).strip():
                parts.append(f"🔁 等价：{r.get('EquivalentCourses')}")
            if str(r.get("ExclusionCourses","")).strip():
                parts.append(f"🚫 互斥：{r.get('ExclusionCourses')}")
            if str(r.get("Description","")).strip():
                parts.append(f"📝 描述：{r.get('Description')}")
            outputs.append("\n".join(parts))
    return safe_answer("\n".join(outputs))

# —— 推荐 / 简单Plan（保留）
CATEGORY_RANK = {
    "Foundational Core Courses": 1,
    "Core Courses": 2,
    "Prescribed Electives": 3,
    "Disciplinary Prescribed Electives": 3,
    "Advanced Disciplinary Electives": 3,
    "Disciplinary Electives": 4,
    "General Education": 5,
    "Capstone": 6,
    "Research": 7,
    "Other": 8,
}
USER_COMPLETED: set = set()

def recommend_course_node(state):
    raw = state["query"]
    query_latest = _extract_latest_user_utterance(raw)
    query_upper = str(query_latest).upper()
    topic = state.get("topic") or "AI"
    num = state.get("num_courses")
    try:
        num = int(num)
    except:
        m = re.search(r"(?:推荐|来|给我|要)?\s*(\d{1,3})\s*(?:门|个)?", query_upper)
        num = int(m.group(1)) if m else 5
    num = max(1, min(num, 50))
    term = None
    for t in ["T1","T2","T3"]:
        if t in query_upper: term = t; break
    cand = df[df["OfferingTerms"].str.upper().str.contains(term, na=False)] if term else df.copy()
    if term: cand = cand[~cand["OfferingTerms"].str.contains("NOT OFFERED", na=False)]
    if cand.empty:
        return safe_answer(f"⚠️ 当前未找到在 {term} 开设的课程。")
    def score_row(r: pd.Series) -> float:
        cat = str(r.get("AutoCategory", r.get("Category","")))
        base = 100 - CATEGORY_RANK.get(cat, 9) * 10
        desc = (((r.get("Description","") + " " + r.get("CourseName",""))).lower())
        topic_hit = 5 if topic.lower() in desc else 0
        penalty = 0
        eq = set(_parse_codes_list(r.get("EquivalentCourses","")))
        ex = set(_parse_codes_list(r.get("ExclusionCourses","")))
        if USER_COMPLETED & (eq | ex): penalty += 50
        cond = (r.get("ConditionsForEnrolment","") or "").lower()
        if re.search(r"prereq|pre-?requisite|requirement|assumed", cond): penalty += 2
        return base + topic_hit - penalty
    cand = cand.copy()
    cand["_score"] = cand.apply(score_row, axis=1)
    cand = cand.sort_values(["_score", "CourseCode"], ascending=[False, True])
    selected = cand.head(num)
    items = []
    for i, r in enumerate(selected.itertuples(index=False), 1):
        short_desc = re.sub(r"\s+", " ", str(getattr(r, "Description", ""))).strip()
        if len(short_desc) > 110: short_desc = short_desc[:110] + "..."
        eq = getattr(r, "EquivalentCourses", ""); ex = getattr(r, "ExclusionCourses", ""); cond = getattr(r, "ConditionsForEnrolment", "")
        tags = []
        cat = getattr(r, "AutoCategory", "") or getattr(r, "Category", "")
        if cat: tags.append(cat)
        if term: tags.append(term)
        meta = " | ".join(tags)
        lines = [
            f"{i}. 📘 {r.CourseCode} - {getattr(r,'CourseName','')} ({getattr(r,'Credits','')}UOC, {getattr(r,'OfferingTerms','')})",
            f"   🏷️ {meta}" if meta else "",
            f"   📝 {short_desc}" if short_desc else "",
            f"   🔑 前置：{cond}" if cond else "",
            f"   🔁 等价：{eq}" if eq else "",
            f"   🚫 互斥：{ex}" if ex else "",
        ]
        items.append("\n".join([x for x in lines if x]))
    header = f"🎯 为你推荐的 {term or topic} 课程（共 {len(items)} 门）："
    return safe_answer([header] + items)

def plan_course_node(state):
    topic = state.get("topic") or "AI"
    selected = df[df["Description"].str.lower().str.contains(topic.lower(), na=False)]
    if selected.empty:
        return safe_answer(f"❌ 未找到与 {topic} 相关的课程。")
    selected["_rank"] = selected["Category"].map(lambda c: CATEGORY_RANK.get(str(c), 9))
    plan = selected.sort_values(["_rank", "CourseCode"]).head(6)
    plan_text = [f"{i+1}. {r.CourseCode} - {r.CourseName} ({r.OfferingTerms})" for i, r in plan.iterrows()]
    return safe_answer([f"🧠 {topic.upper()} 方向建议修读顺序（优先核心）："] + plan_text)

# === Explain Panel Helpers ===
def _lookup_course(code: str) -> pd.Series | None:
    cc = str(code or "").upper().strip()
    hit = df[df["CourseCode"].str.upper()==cc]
    return hit.iloc[0] if not hit.empty else None

def _explain_course_line(code: str) -> str:
    r = _lookup_course(code)
    if r is None: return f"{code}"
    cat = (r.get("AutoCategory") or r.get("Category") or "").strip()
    term = str(r.get("OfferingTerms","")).strip()
    cond = str(r.get("ConditionsForEnrolment","")).strip()
    eq = str(r.get("EquivalentCourses","")).strip()
    ex = str(r.get("ExclusionCourses","")).strip()
    bits = [f"{code}"]
    if cat: bits.append(f"类别:{cat}")
    if term: bits.append(f"开课:{term}")
    if cond: bits.append(f"先修:{cond}")
    if eq: bits.append(f"等价:{eq}")
    if ex: bits.append(f"互斥:{ex}")
    return "；".join(bits)

def build_explain_panel(schedule: dict[int, list[str]], terms: list[str], state: dict) -> str:
    lines = []
    # 偏好回显
    ex_codes = [c.upper() for c in state.get("exclude_codes", [])]
    ex_topics = list(state.get("exclude_topics", []))
    if ex_codes or ex_topics:
        parts = []
        if ex_codes: parts.append("排除课程：" + "、".join(ex_codes))
        if ex_topics: parts.append("排除主题：" + "、".join(ex_topics))
        lines.append("🧠 偏好记忆：" + "；".join(parts))
    # 每学期解释
    for i, term in enumerate(terms, 1):
        xs = schedule.get(i, [])
        if not xs: continue
        lines.append(f"• {term}:")
        for code in xs:
            lines.append("   - " + _explain_course_line(code))
    # 引用：来自 CSV 数据源
    lines.append("🔗 引用：课程名称/先修/等价/互斥/开课学期均来自本地 CSV（" + os.path.basename(CSV_FILE) + "）。")
    return "\n".join(lines)
def grad_plan_node(state):
    global LAST_PLAN_SCHEDULE, LAST_PLAN_TERMS, LAST_PLAN_STATE
    # On-demand Explain: only show explanation for the last generated plan
    if state.get("explain_only"):
        if LAST_PLAN_SCHEDULE is None or LAST_PLAN_TERMS is None:
            return safe_answer("📝 还没有可解释的计划。请先让我生成一次选课建议，再说“请给解释”。")
        explain = build_explain_panel(LAST_PLAN_SCHEDULE, LAST_PLAN_TERMS, LAST_PLAN_STATE)
        return safe_answer("🧾 解释与引用\n" + explain)
    raw = state.get("query","")
    latest = _extract_latest_user_utterance(raw)

    # 已修课程：支持 COMP9021 / 裸数字 9021 => COMP9021
    completed = _parse_completed_codes(latest)

    # 学期负载：若用户写了 "332 332" 之类，就用之；否则交给排课器默认
    term_loads = _parse_term_loads(latest)

    prefer = state.get("route_pref","auto")
    schedule, terms, missing, route_type = _schedule_ai_plan(
        df, AI_RULES, completed=completed, prefer_route=prefer, term_loads=term_loads,
        exclude_codes=set(state.get("exclude_codes", set())), exclude_topics=set(state.get("exclude_topics", set()))
    )

    lines = ["📚 AI方向两年学习建议（草案）"]
    for i, t in enumerate(terms, 1):
        xs = schedule[i]
        lines.append(f"{i}. {t}： " + ("，".join(xs) if xs else "（暂未安排）"))

    if missing:
        name_map = {"found_core": "基础核心课","adv_core": "高级核心课","ai_core": "AI核心课","dke": "DKE","elective": "一般选修","project": "毕业研究/项目","noncap_total": "非Capstone学分总量"}
        warn = "；".join([f"{name_map.get(k,k)} 仍缺约 {u} 学分" for k,u in missing])
        lines.append(f"\n⚠️ 覆盖提示：{warn}。可补充同类课程或调整学期以满足毕业标准。")

    if route_type != "未确定":
        lines.append(f"\n🏁 当前规划路线：{route_type}")

    if term_loads:
        lines.append(f"\n🧩 学期负载采用：{'-'.join(map(str, term_loads[:3]))} / {'-'.join(map(str, term_loads[3:]))}（每学期最多3门，最少1门）")

    lines.append("\n🔎 说明：优先安排 3 门基础核心课，其余（高级核心/AI核心/DKE/一般选修）可穿插；"
                 "Capstone/Research 在先修允许时尽量安排到后两学期；严格遵守 CSV 的开课学期。")

    # Cache the latest plan for on-demand explanation
    LAST_PLAN_SCHEDULE = schedule
    LAST_PLAN_TERMS = terms
    LAST_PLAN_STATE = dict(state)
    return safe_answer("\n".join(lines))


def _extract_codes_from_text(s: str) -> list[str]:
    import re as _re
    text = str(s or "")

    # 1) 先抓完整课号（容忍空格/连字符）：COMP9414 / COMP-9414 / COMP 9414
    full = _re.findall(r"(?<![A-Z0-9])([A-Z]{4}\s*-?\s*\d{4})(?![A-Z0-9])", text.upper())
    full = [_re.sub(r"[\s-]+","",c) for c in full]

    # 2) 再抓 4 位数字简写：9414 / 9814
    short = _re.findall(r"(?<!\d)(\d{4})(?!\d)", text)
    mapped = []
    for d in short:
        try:
            # 在 df 里找“课程代码以该 4 位结尾”的候选
            candidates = df[df["CourseCode"].str.endswith(d)]["CourseCode"].str.upper().unique().tolist()
        except Exception:
            candidates = []
        # 有 COMP 优先取 COMP；否则唯一就取唯一；多于一个就先取第一个（需要的话可做消歧提示）
        comp_first = [c for c in candidates if c.startswith("COMP")]
        if comp_first:
            mapped.append(comp_first[0])
        elif len(candidates) == 1:
            mapped.append(candidates[0])
        elif len(candidates) > 1:
            mapped.append(candidates[0])

    # 3) 合并去重（保持顺序）
    out, seen = [], set()
    for c in full + mapped:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def reviews_node(state):
    q = _extract_latest_user_utterance(state.get("query",""))
    codes = _extract_codes_from_text(q)
    import re as _re
    # 对比场景
    if len(codes) >= 2 and _re.search(r"(对比|区别|diff|compare|哪个好|更推荐|vs|比较)", q.lower()):
        a, b = codes[:2]
        return safe_answer(compare_reviews(a, b))
    # 单课评价
    if len(codes) >= 1:
        return safe_answer(summarize_reviews_for(codes[0]))
    # 没抓到课程号：提示用户提供
    return safe_answer("想看哪门课的口碑？请带上课程号（如 COMP9414）。\n例如：COMP9414 评价怎么样？")

# === Export Node ===
def export_node(state):
    path = state.get("export_path") or "plan.csv"
    try:
        _has = (LAST_PLAN_SCHEDULE is not None) and (LAST_PLAN_TERMS is not None)
    except NameError:
        _has = False
    if not _has:
        return safe_answer("📝 还没有可导出的计划。请先让我生成一次选课建议，再说“导出计划”。")
    try:
        out = export_plan_csv(LAST_PLAN_SCHEDULE, LAST_PLAN_TERMS, df, out_path=path, include_desc=True)
        return safe_answer(f"✅ 已导出：{out}\n包含列：Term, CourseCode, CourseName, Description")
    except Exception as e:
        return safe_answer(f"⚠️ 导出失败：{e}")

# === Export ICS Node ===
def export_ics_node(state):
    path = state.get("export_path") or "plan.ics"
    try:
        _has = (LAST_PLAN_SCHEDULE is not None) and (LAST_PLAN_TERMS is not None)
    except NameError:
        _has = False
    if not _has:
        return safe_answer("📝 还没有可导出的计划。请先让我生成一次选课建议，再说“导出日历”。")
    try:
        out = export_plan_ics(LAST_PLAN_SCHEDULE, LAST_PLAN_TERMS, df, out_path=path)
        return safe_answer(f"✅ 已导出：{out}\n用法：在 Google/Apple/Outlook 日历中导入 .ics 即可（All-day 事件，每门课一条）。")
    except Exception as e:
        return safe_answer(f"⚠️ 导出失败：{e}")

# =========================================================
# Router（含“研究/项目路线”偏好、描述识别）
# =========================================================
def router_node(state):
    q = _extract_latest_user_utterance(state.get("query", ""))
    q_low = q.lower()
    if re.search(r"(导出|保存).*(计划|csv)|\bexport\b.*\bcsv\b|导出csv|导出课程表", q_low):
        return {"next_node": "export"}
    if re.search(r"(导出|保存).*(日历|ics)|\bexport\b.*\bics\b|导出日历|导出ics", q_low):
        return {"next_node": "export_ics"}
    # 课程评价/口碑/难度/工作量/对比 → reviews
    _codes_in_q = re.findall(r"(?<![A-Z0-9])([A-Z]{4}\d{4})(?![A-Z0-9])", q.upper())
    if re.search(r"(评价|口碑|测评|review|reviews|难度|workload|作业多不多|作业|哪个好|更推荐|对比|区别|比较|vs)", q_low):
        return {"next_node": "reviews"}
    # On-demand explain trigger
    if re.search(r"(给.*解释|请给解释|解释与引用|^解释$|解释一下|解释下|explain|why|为什么)", q_low):
        return {"next_node": "grad_plan", "explain_only": True, "route_pref": state.get("route_pref","auto")}

    # 显式路线偏好
    route_pref = "auto"
    # 支持课程号/同义词触发
    if re.search(r"(research|thesis|9991|9992|9993|研究|论文)", q_low):
        route_pref = "research"
    if re.search(r"(project|capstone|9900|gsoe9010|gsoe9011|项目|毕设|课设)", q_low):
        route_pref = "project"
    # 解析排除偏好（课程号/主题）
    ex_codes, ex_topics = parse_exclusions(q)

    # 学位规划/选课建议
    if re.search(r"毕业|标准|学位|方案|路径|规划|学习计划|两年|选课建议|study\s*plan|degree\s*plan|roadmap", q_low):
        return {"detail_type": "", "route_pref": route_pref, "exclude_codes": ex_codes, "exclude_topics": ex_topics, "next_node": "grad_plan"}

    detail_type = ""
    if re.search(r"前置|先修|prereq|pre-?requisite|requirement|enrolment", q_low):
        detail_type = "prereq"
    elif re.search(r"互斥|排斥|exclusion", q_low):
        detail_type = "exclusion"
    elif re.search(r"等价|相当|equivalent", q_low):
        detail_type = "equivalent"
    elif re.search(r"类型|类别|category|属于什么类型", q_low):
        detail_type = "category"
    elif re.search(r"描述|简介|介绍|内容|大纲|syllabus|description|overview|about|是什么|what\s+is|讲什么|学什么", q_low):
        detail_type = "desc"

    if detail_type:
        return {"detail_type": detail_type, "route_pref": route_pref, "next_node": "detail"}
    else:
        return {"route_pref": route_pref, "next_node": "detect"}

# =========================================================
# Graph
# =========================================================
class CourseState(TypedDict):
    explain_only: bool
    query: str
    intent: str
    course_code: str
    topic: str
    num_courses: Any
    detail_type: str
    route_pref: str
    next_node: str
    answer: Annotated[List, add_messages]

graph = StateGraph(CourseState)
graph.add_node("router", router_node)
graph.add_node("detect", detect_intent_node)
graph.add_node("chitchat", chitchat_node)
graph.add_node("search", search_course_node)
graph.add_node("term_query", term_query_node)
graph.add_node("detail", detail_query_node)
graph.add_node("recommend", recommend_course_node)
graph.add_node("plan", plan_course_node)
graph.add_node("grad_plan", grad_plan_node)
graph.add_node("export", export_node)
graph.add_node("export_ics", export_ics_node)
graph.add_node("reviews", reviews_node)

graph.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {"detail": "detail", "grad_plan": "grad_plan", "reviews": "reviews", "export": "export", "export_ics": "export_ics", "detect": "detect"},
)
graph.add_conditional_edges(
    "detect",
    lambda state: state["next_node"],
    {"chitchat": "chitchat", "search": "search", "term_query": "term_query",
     "recommend": "recommend", "plan": "plan", "grad_plan": "grad_plan"},
)
for node in ["chitchat","search","term_query","detail","recommend","plan","grad_plan","reviews","export","export_ics"]:
    graph.add_edge(node, END)
graph.set_entry_point("router")
app = graph.compile()

# === Web 前端入口：单轮应答 ===
def agent_respond(user_text: str) -> str:
    """
    输入一段用户文本，返回本轮答案字符串。
    - 复用已编译的 graph（app）与全局缓存（LAST_PLAN_*）。
    """
    try:
        compiled = globals().get("app", None)
        if compiled is None:
            compiled = graph.compile()
            globals()["app"] = compiled
        out = compiled.invoke({"query": str(user_text)})
        if isinstance(out, dict) and "answer" in out:
            # 将消息数组拼接成文本
            msgs = out["answer"]
            parts = []
            for a in msgs:
                try:
                    parts.append(a.content if hasattr(a, "content") else str(a))
                except Exception:
                    parts.append(str(a))
            return "\n".join(parts).strip()
        return str(out)
    except Exception as e:
        return f"⚠️ 出错了：{e}"


# =========================================================
# Main loop
# =========================================================
if __name__ == "__main__":
    print("\n🎓 UNSW Course Advisor Agent 启动成功！")
    #print("📄 使用数据:", CSV_FILE)
    print("💬 hi 你可以问我关于选课的各类问题哦~ \n")
    print("问完问题 输入exit退出哦~ \n")

    context_memory = ""
    while True:
        q = input("💬 请输入问题：")
        if q.lower() in ["exit","quit","q"]:
            print("👋 再见！"); break
        try:
            full_query = context_memory + f"\n用户：{q}"
            result = app.invoke({"query": full_query})
            all_answers = result.get("answer", [])
            output_lines = [a.content if hasattr(a, "content") else str(a) for a in all_answers]
            print("\n🤖 回复：\n" + "\n".join(output_lines) + "\n")
            context_memory += f"\n用户：{q}\n助手：{output_lines[-1] if output_lines else ''}"
            context_memory = "\n".join(context_memory.splitlines()[-8:])
        except Exception as e:
            print(f"⚠️ 出现错误：{e}\n")
