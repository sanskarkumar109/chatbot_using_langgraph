import os

import streamlit as st

from langchain_core.messages import HumanMessage

from graph import graph

from rag import build_vectorstore


st.set_page_config(
    page_title="AI Research Agent"
)

st.title(
    "AI Research Agent"
)

# --------------------
# PDF Upload
# --------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = (
        f"data/uploaded_pdfs/"
        f"{uploaded_file.name}"
    )

    with open(
        save_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner(
        "Indexing PDF..."
    ):

        build_vectorstore(
            save_path
        )

    st.success(
        "PDF Indexed Successfully"
    )

# --------------------
# Chat History
# --------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.write(
            msg["content"]
        )

# --------------------
# Chat Input
# --------------------

prompt = st.chat_input(
    "Ask Anything..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message(
        "user"
    ):
        st.write(prompt)

    config = {
        "configurable": {
            "thread_id": "research_agent"
        }
    }

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=prompt
                )
            ]
        },
        config=config
    )

    answer = (
        response["messages"][-1]
        .content
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message(
        "assistant"
    ):
        st.write(answer)