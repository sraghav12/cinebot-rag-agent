import streamlit as st
import traceback

from config import setup_environment
from graph import get_app

st.set_page_config(
    page_title="Agent Explorer",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LangGraph AstraDB AI Agent")
st.markdown("A modular RAG agent built with **LangGraph**, **AstraDB**, and **Groq**.")

# Initialize the system parameters once
@st.cache_resource
def initialize_agent():
    try:
        setup_environment()
        return get_app()
    except Exception as e:
        st.error(f"Failed to initialize the agent. Error: {str(e)}")
        return None

app = initialize_agent()

if app is None:
    st.stop()
    
# Store conversational history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about agents, adversarial attacks, or anything else!"):
    # Append the user's message to the history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # We'll stream the node steps so the user sees some intermediate process
        with st.status("Agent thinking...", expanded=True) as status:
            try:
                inputs = {"question": prompt}
                final_results = None
                
                # Stream nodes from LangGraph
                for output in app.stream(inputs):
                    for key, value in output.items():
                        st.write(f"🔄 Routing through node: **`{key}`**")
                        final_results = value
                
                # Parse final answer
                docs = final_results.get('documents', [])
                if isinstance(docs, list) and len(docs) > 0:
                    answer = docs[0].metadata.get('description', docs[0].page_content)
                elif hasattr(docs, 'page_content'):
                    # Useful for Wikipedia results which might be a single Document object
                    answer = docs.page_content
                else:
                    answer = str(docs)

                status.update(label="Response generated!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="An error occurred", state="error", expanded=True)
                st.error(f"Execution failed: {traceback.format_exc()}")
                answer = "I'm sorry, I encountered an error while processing your request."

        # Display final response inside the message column
        message_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
