import streamlit as st
import main_summarized_tickets_final
import os
from docx import Document  # Import Document from python-docx for Word document generation

# Define Streamlit app
def debug():
    st.title("Chatbot for Tickets Support")

    # Sidebar for user input and session control
    session_id = st.text_input("Enter Session ID:", key="session_id")
    clear_history = st.button("Clear Chat History")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    # Initialize last_question and last_response with default values
    last_question = "No question available"
    last_response = "No response available"

    # Clear chat history if requested
    if clear_history:
        st.session_state.chat_history.clear()

    # Sidebar for user input
    user_input = st.text_input("Enter your message:")

    if st.button("Send"):
        # Add user input to chat history
        if st.session_state.session_id not in st.session_state.chat_history:
            st.session_state.chat_history[st.session_state.session_id] = []
        st.session_state.chat_history[st.session_state.session_id].append(("You:", user_input))

        # Process user input and generate response
        response = process_user_input(user_input, st.session_state.session_id)
        
        # Update last_question and last_response
        last_question = user_input
        last_response = response

        # Add bot response to chat history
        st.session_state.chat_history[st.session_state.session_id].append(("Bot:", response))

    # Display last question and its response
    if st.session_state.session_id in st.session_state.chat_history:
        for entry in reversed(st.session_state.chat_history[st.session_state.session_id]):
            if last_response == "No response available" and entry[0] == "You:":
                last_question = entry[1]
            elif last_question != "No question available" and entry[0] == "Bot:":
                last_response = entry[1]
                break
        if last_question != "No question available" and last_response != "No response available":
            st.subheader("Last Question and Its Response")
            st.write("Question:", last_question)
            st.write("Response:", last_response)

    # Button to generate email with response to last question
    if st.button("Generate Email with Response"):
        generate_email_with_response(last_question, last_response)

    # Display chat history in the sidebar
    selected_session_id = st.sidebar.selectbox("Select Session ID:", list(st.session_state.chat_history.keys()))
    st.sidebar.subheader(f"Chat History for Session {selected_session_id}")
    if selected_session_id:
        for entry in st.session_state.chat_history[selected_session_id]:
            st.sidebar.write(entry[0], entry[1])

    # Button to export chat history to Word document
    if st.sidebar.button("Export Chat History to Word"):
        export_chat_history_to_word(selected_session_id)

# Process user input and generate response
def process_user_input(user_input, session_id):
    global conversational_rag_chain

    conversational_rag_chain = main_summarized_tickets_final.conversational_rag_chain

    # Re-rank documents using the updated function
    re_ranked_docs = main_summarized_tickets_final.get_re_ranked_documents(user_input)
    context = "\n".join([doc.page_content for doc in re_ranked_docs])
    
    # Call the LangChain pipeline with user input and re-ranked context
    response = conversational_rag_chain.invoke(
        {
            'input': user_input,
            'chat_history': [],
            'context': context
        },  # Pass user input as input
        {'configurable': {'session_id': session_id}}
    ) # Add configuration

    # Retrieve the generated response from the output key "answer"
    generated_response = response.get("answer", "No response found")  # Provide default if missing

    # Return the generated response for Streamlit to display
    return generated_response

# Function to generate email with response to last question
def generate_email_with_response(question, response):
    # You can implement the logic to generate the email here
    st.write(f"Dear [Recipient],\n\nIn response to your query, \n\n{response}\n\nRegards,\n")

# Function to export chat history to Word document
def export_chat_history_to_word(session_id):
    # Create a new Word document
    doc = Document()

    # Add chat history to Word document
    for entry in st.session_state.chat_history[session_id]:
        # Add entry to the document
        doc.add_paragraph(f"{entry[0]} {entry[1]}")

    # Save Word document
    word_file_path = f"chat_history_session_{session_id}.docx"
    doc.save(word_file_path)
    st.sidebar.success(f"Chat history for Session {session_id} exported to {word_file_path}")

if __name__ == "__main__":
    debug()
