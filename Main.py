import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fitz  
from docx import Document
import nltk
import re

nltk.download('punkt')

# Load the summarization model
summarization_model_name = "Legal_Pegasus/"
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

# Load LegalBERT for Q&A
qa_model_name = "LegalBERT/"
qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=64,  
        min_length=50, 
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        temperature=0.7,
    )
    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    cleaned_summary = clean_summary(raw_summary)
    return cleaned_summary


def clean_summary(summary):
    
    if not summary.endswith(('.', '!', '?')):
        summary = re.sub(r'[^.!?]*$', '', summary)
        summary = summary.strip()
    return summary

def chunk_text_by_sentences(text, max_tokens=1024):
    sentences = nltk.sent_tokenize(text)
    tokenized_chunks = []
    current_chunk = ""
    current_chunk_len = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence)
        if current_chunk_len + len(tokenized_sentence) <= max_tokens:
            current_chunk += " " + sentence
            current_chunk_len += len(tokenized_sentence)
        else:
            tokenized_chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_len = len(tokenized_sentence)

    if current_chunk: 
        tokenized_chunks.append(current_chunk.strip())

    return tokenized_chunks

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        uploaded_file = "Legal Documents//"+uploaded_file.name
        pdf_document = fitz.open(uploaded_file)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        uploaded_file = "Legal Documents//"+uploaded_file.name
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    return ""

# Initialize session state
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_summary" not in st.session_state:
    st.session_state.final_summary = None

st.title("Legal Document Summarizer and Q&A Assistant")
uploaded_file = st.file_uploader("Upload legal document (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    
    with st.spinner("Extracting text from the uploaded file..."):
        text = extract_text_from_file(uploaded_file)

    if text:
        st.success("Text extraction complete!")
        st.session_state.document_text = text

        if st.session_state.final_summary is None:

            chunks = chunk_text_by_sentences(text)

            st.subheader("Summarizing the Document...")
            progress_bar = st.progress(0)
            chunk_summaries = []

            for i, chunk in enumerate(chunks):
                chunk_summary = summarize(chunk)
                chunk_summaries.append(chunk_summary)
                progress = (i + 1) / len(chunks)
                progress_bar.progress(progress)

            st.session_state.final_summary = " ".join(chunk_summaries).strip()
            final_summary = clean_summary(st.session_state.final_summary)
            progress_bar.empty()

        st.subheader("Summary:")
        st.write(st.session_state.final_summary)

        # Chat Section
        st.subheader("Chat with the Document:")
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Type your question:")
            submit = st.form_submit_button("Ask")

        if submit and question.strip():
            with st.spinner("Processing your question..."):

                answer = qa_pipeline(question=question, context=st.session_state.document_text)
                st.session_state.chat_history.append({"question": question, "answer": answer['answer']})
            st.success("Answer generated!")

        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}: {chat['question']}**")
                st.write(f"**A{i+1}: {chat['answer']}**")
    else:
        st.error("Unable to extract text from the uploaded file.")
