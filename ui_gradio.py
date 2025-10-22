from __future__ import annotations
import os
import gradio as gr

import UNSW_Course_Agent as agent

# ====== 基本信息 ======
TITLE = "UNSW Course Agent"
TAGLINE = "By Jiawei Sun"

DESC = (
    """
**示例指令**
- 给我AI两年选课建议 我要project 233 233 不要9414
- 请给解释
- 9414 vs 9814 哪个更推荐
- COMP9414 最近评价怎么样
- 导出计划 / 导出日历
    """
)

# ====== Agent 包装 ======

def respond(message: str, history):
    """
    修复了没有记忆
    """
    def hist_to_transcript(hist) -> str:
        txt = ""
        for h in (hist or []):
            # 兼容多种 gradio 版本：tuple、dict、Message 对象
            if isinstance(h, (list, tuple)) and len(h) == 2:
                u, b = h
                if u: txt += f"\n用户：{u}"
                if b: txt += f"\n助手：{b}"
            else:
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else None)
                content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else None)
                if role == "user" and content is not None:
                    txt += f"\n用户：{content}"
                elif role in ("assistant", "bot") and content is not None:
                    txt += f"\n助手：{content}"
        return txt

    transcript = hist_to_transcript(history)
    full_query = (transcript + f"\n用户：{message}").strip()
    return agent.agent_respond(full_query)

def do_export_csv():
    msg = agent.agent_respond("导出计划")
    path = "plan.csv"
    return msg, (path if os.path.exists(path) else None)


def do_export_ics():
    msg = agent.agent_respond("导出日历")
    path = "plan.ics"
    return msg, (path if os.path.exists(path) else None)


# ====== Theme & CSS ======
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

CSS = """
#header-card {border-radius: 16px; padding: 16px 18px; background: linear-gradient(135deg,#eef2ff, #e0e7ff);}
#title {text-align:center; font-weight:800; font-size: 28px; letter-spacing: .3px; margin: 6px 0 4px;}
#tagline {text-align:center; color:#475569; margin: 0 0 10px;}
#desc {max-width: 900px; margin: 0 auto; color:#334155;}
#toolbar {margin-top: 8px; display:flex; justify-content:center; gap:10px;}
#export-area {border: 1px dashed #c7d2fe; border-radius: 14px; padding: 14px;}
.footer {text-align:center; color:#94a3b8; font-size: 12px; margin-top: 8px;}
@media (max-width: 640px){ #title{font-size:22px;} }
"""

# ====== UI ======
with gr.Blocks(title=TITLE, theme=THEME, css=CSS, fill_height=True) as demo:
    with gr.Column():
        with gr.Group(elem_id="header-card"):
            gr.Markdown("""
<p id=title>🎓 UNSW Course Agent</p>
<p id=tagline>by Jiawei SUN</p>
<div id=toolbar>
</div>
""", elem_id="title-area")
        gr.Markdown(DESC, elem_id="desc")

    chat = gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=[
            "给我AI两年选课建议 我要project 233 233 不要9414",
            "请给解释",
            "9414 vs 9814 哪个更推荐",
            "COMP9414 最近评价怎么样",
            "导出计划",
            "导出日历",
        ],
    )

    with gr.Accordion("📤 导出计划 / Export", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    csv_btn = gr.Button("导出 CSV", variant="primary", size="lg")
                    ics_btn = gr.Button("导出 ICS", variant="secondary", size="lg")
                out_msg = gr.Textbox(label="导出结果", interactive=False, lines=3)
            with gr.Column(scale=2, elem_id="export-area"):
                file_csv = gr.File(label="下载 plan.csv", visible=False)
                file_ics = gr.File(label="下载 plan.ics", visible=False)

    gr.Markdown("""
<div class=footer>Made with ❤️ Gradio · 如果按钮不可用，请先在聊天区生成计划后再试</div>
""")

    # ====== Events ======
    def wrap_csv():
        msg, fpath = do_export_csv()
        return msg, (gr.update(visible=bool(fpath), value=fpath))

    def wrap_ics():
        msg, fpath = do_export_ics()
        return msg, (gr.update(visible=bool(fpath), value=fpath))

    csv_btn.click(fn=wrap_csv, outputs=[out_msg, file_csv])
    ics_btn.click(fn=wrap_ics, outputs=[out_msg, file_ics])

# ====== 启动 ======
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
