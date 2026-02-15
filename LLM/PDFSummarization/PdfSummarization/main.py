import streamlit as st
import os 
import langchain
from metier import summarizer 
def main():
    st.set_page_config(page_title="PDF Summurization")
    st.title("PDF Summarizing App")
    st.write("Summarize your PDF file in just in few seconds !")
    st.divider()
    file_uploaded = st.file_uploader("Upload your PDF Document.",type="pdf")
    btn = st.button("Generate Summary")
    resp = summarizer(file_uploaded)
    if btn : 
        st.text(resp)
if __name__ == '__main__' :
    main()
   
