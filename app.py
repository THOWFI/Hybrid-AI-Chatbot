import json
import time
import streamlit as st

from scripts.router import choose_model, refresh_keywords
from scripts.llm_utils import query_model, stream_model
from scripts.nlp_enhancer import clean_text, enhance_text
from scripts.scrape_keywords import scrape_many, merge_into_file, DEFAULT_PAGES

# -----------------------------
# Page + minimal styling
# -----------------------------
st.set_page_config(page_title="ARSHU - Hybrid AI Chatbot", layout="centered")

st.markdown(
    """
    <style>
      /* Chat bubbles */
      .bubble-user   { text-align:right;  background:#2b7cff; color:#fff; padding:10px 14px; border-radius:14px; display:inline-block; max-width:80%; word-wrap:break-word; }
      .bubble-bot    { text-align:left;   background:#2f2f2f; color:#fff; padding:10px 14px; border-radius:14px; display:inline-block; max-width:80%; word-wrap:break-word; }
      .chat-row      { margin: 8px 0 2px; }
      .caption-row   { margin: 0 0 12px; opacity:0.7; font-size:0.85rem; }
      footer { visibility: hidden; } /* hide streamlit footer */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Hybrid LLM Chatbot")
st.caption("Chat naturally with **Gemma-2B** (casual/basic) or **Mistral-7B** (technical).")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.subheader("Maintenance")
    if st.button("Scrape Wikipedia (default topics)"):
        with st.spinner("Scraping Wikipedia…"):
            kws = scrape_many(DEFAULT_PAGES)
            merge_into_file(kws)
        st.success("Scraped & merged keywords.")

    if st.button("Reload Keywords Only"):
        refresh_keywords()
        st.success("Reloaded keywords into router.")

    st.subheader("Settings")
    stream_mode = st.radio(
        "Response Mode",
        ["Streaming (ChatGPT-style)", "Full reply"],
        index=0,
    )

    model_selection = st.selectbox(
        "Model Selection",
        ["Auto", "Gemma", "Mistral"],
        index=0,
        help="Auto = keyword routing (Gemma for casual, Mistral for technical)."
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.30, 0.05,
                            help="Higher = more creative. Lower = more factual.")
    max_tokens = st.slider("Max tokens per reply", 64, 1024, 512, 32)

    memory_turns = st.slider(
        "Conversation memory (turns)",
        0, 8, 4,
        help="How many previous user+assistant pairs to include as context."
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    with cols[1]:
        if st.button("Export chat"):
            ts = int(time.time())
            fname = f"chat_export_{ts}.json"
            st.download_button(
                label="Download JSON",
                data=json.dumps(st.session_state.get("chat_history", []), ensure_ascii=False, indent=2),
                file_name=fname,
                mime="application/json"
            )


# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# -----------------------------
# Helpers
# -----------------------------
def render_message(role: str, content: str, model: str = None):
    """Render one message with bubble or code box."""
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"<div class='chat-row'><span class='bubble-user'>{content}</span></div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # detect fenced code block (```lang ... ```)
            if content.strip().startswith("```"):
                # try to extract language and inner code
                lines = content.split("\n")
                fence = lines[0].strip("`")
                lang = fence if fence else None
                # inner code may occupy all following lines up to last fence (if present)
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    code = "\n".join(lines[1:-1])
                else:
                    # if no closing fence yet, show all after first line
                    code = "\n".join(lines[1:])
                st.code(code, language=lang or "text")
            else:
                st.markdown(f"<div class='chat-row'><span class='bubble-bot'>{content}</span></div>", unsafe_allow_html=True)

            if model:
                st.markdown(f"<div class='caption-row'>Model used: <b>{model.capitalize()}</b></div>", unsafe_allow_html=True)


# -----------------------------
# Render history
# -----------------------------
def render_history():
    for msg in st.session_state["chat_history"]:
        render_message(msg["role"], msg["content"], msg.get("model"))


# 1) Render everything that already happened
render_history()

# 2) Input at the very bottom
user_input = st.chat_input("Type your message…")

if user_input:
    # 2a) Append and render the new user message immediately
    user_input_clean = enhance_text(clean_text(user_input))
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    render_message("user", user_input)

    # 2b) Decide model
    if model_selection == "Auto":
        model_choice = choose_model(user_input_clean)
    else:
        model_choice = model_selection.lower()

    # 2c) Short memory
    hist = st.session_state["chat_history"][-(2 * memory_turns):] if memory_turns > 0 else []

    # 2d) Generate reply
    response_text = ""

    if stream_mode.startswith("Streaming"):
        # streaming yields cleaned cumulative buffer strings
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            try:
                for chunk in stream_model(
                    model_choice,
                    user_input_clean,
                    history=hist,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    # chunk is the current full cleaned output (not a single token)
                    response_text = chunk  # replace (not accumulate)
                    # Live preview with code detection
                    if response_text.strip().startswith("```"):
                        lines = response_text.split("\n")
                        fence = lines[0].strip("`")
                        lang = fence if fence else None
                        if len(lines) >= 3 and lines[-1].strip() == "```":
                            code = "\n".join(lines[1:-1])
                        else:
                            code = "\n".join(lines[1:])
                        placeholder.code(code, language=lang or "text")
                    else:
                        placeholder.markdown(
                            f"<div class='chat-row'><span class='bubble-bot'>{response_text}▌</span></div>",
                            unsafe_allow_html=True
                        )
            except Exception as e:
                response_text = f"⚠️ Error while streaming from {model_choice}: {e}"
        # 🔥 Finalize without the ▌ cursor
        if response_text.strip().startswith("```"):
            lines = response_text.split("\n")
            fence = lines[0].strip("`")
            lang = fence if fence else None
            if len(lines) >= 3 and lines[-1].strip() == "```":
                code = "\n".join(lines[1:-1])
            else:
                code = "\n".join(lines[1:])
            placeholder.code(code, language=lang or "text")
        else:
            placeholder.markdown(
                f"<div class='chat-row'><span class='bubble-bot'>{response_text}</span></div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<div class='caption-row'>Model used: <b>{model_choice.capitalize()}</b></div>",
            unsafe_allow_html=True
        )

    else:
        try:
            response_text = query_model(
                model_choice,
                user_input_clean,
                history=hist,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            response_text = f"⚠️ Error while generating with {model_choice}: {e}"
        render_message("assistant", response_text, model_choice)

    # 2e) Persist
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": response_text, "model": model_choice}
    )

# 3) Auto-scroll
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
