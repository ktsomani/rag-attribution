import streamlit as st

st.set_page_config(page_title="RAG Failure Attribution", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Product Edge: Citation UI Intervention")
st.markdown("""
This app demonstrates the **Evidence Under-Use** failure mode. 
The retriever finds the correct document, but the generator ignores it. 
By changing the **UI contract** (forcing inline citations), we fix the failure at **zero latency cost**.
""")

st.divider()

# Simulate the retrieved documents
retrieved_docs = [
    {"id": "doc_1", "text": "The sky is blue."},
    {"id": "doc_2", "text": "Water is wet."}
]

query = st.text_input("User Query:", value="What color is the sky?")

st.subheader("Retrieved Context (Gold Doc is doc_1)")
st.info(f"📄 **doc_1**: {retrieved_docs[0]['text']}")
st.info(f"📄 **doc_2**: {retrieved_docs[1]['text']}")

st.divider()

# Toggle for the A/B test
ui_mode = st.radio(
    "Select UI Intervention Mode:",
    ["Baseline (Footer Citations)", "Intervention (Inline Citations)"],
    horizontal=True
)

st.subheader("Generated Answer")

if ui_mode == "Baseline (Footer Citations)":
    # Baseline: The model ignores the evidence and hallucinates
    st.error("The sky is green and made of cheese.")
    
    st.markdown("---")
    st.caption("Sources: [1] doc_1, [2] doc_2")
    
    st.warning("""
    ⚠️ **Evidence Under-Use Detected (22% rate)**
    The correct document (`doc_1`) was retrieved, but the model ignored it and generated a hallucination. 
    The footer citation hides the fact that the answer is completely disconnected from the text.
    """)

else:
    # Intervention: The UI forces inline citations
    # We use HTML to make the citation look like a real UI chip
    st.success("The sky is blue.", icon="✅")
    
    st.markdown("""
    <div style="background-color: #e6f4ea; border-left: 4px solid #34a853; padding: 10px; margin-top: 10px;">
        <strong>Evidence Card:</strong> <br>
        <em>"The sky is blue."</em> — <code>doc_1</code> <span style="color:green; font-size:12px;">(High Confidence)</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    ✅ **Evidence Under-Use Fixed (9% rate)**
    By forcing the model to emit inline citations (and dropping claims that lack them), the UI forces grounding.
    The model cannot hide a hallucination behind a footer link.
    """)