import nltk
import os
import os

nltk.download('stopwords')
from streamlit_option_menu import option_menu
from collections import Counter
import re
import pandas as pd
import PyPDF2, pdfplumber, nlp, re, docx2txt, streamlit as st, nltk
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
# from nltk.corpus import stopwords
from pathlib import Path
import json
from pyresparser import ResumeParser
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from multiprocessing.pool import ThreadPool
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
import time

nltk.download('punkt')

cities_data = open("./cities.txt", 'r').readlines()


def get_knowledge_base(embeddings, text):
    """
    Generates a knowledge base using text embeddings.

    :param embeddings: Embeddings of the text
    :param text: The text to be processed.
    :return knowledgeBase (FAISS): A knowledge base constructed from the text.
    """
    api_key = st.secrets['api_key']
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase


def get_details_from_openai(text, query, llm, knowledgeBase):
    """
    Retrieves details based on a query using OpenAI's language model and a knowledge base.

    :param text: Input text to be processed.
    :param query: The query for retrieving details.
    :param llm: OpenAI's language model instance.
    :param knowledgeBase: Knowledge base constructed from text embeddings.
    :return: Details retrieved based on the query.
    """
    api_key = st.secrets['api_key']
    docs = knowledgeBase.similarity_search(query)
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=query)
    return response


def get_email_addresses(string):
    """
    Extracts email addresses from a given resume string.

    :param string: The input resume string from which email addresses are to be extracted.
    :return: A comma-separated string of unique email addresses found in the input string.
    """
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return ','.join(list(set(r.findall(string))))


def get_phone_numbers(string):
    """
    Extracts phone numbers from a given resume string.

    :param string: The input resume string from which phone numbers are to be extracted.
    :return: A comma-separated string of extracted phone numbers found in the input string.
    """
    nlp = spacy.load("en_core_web_sm")
    phone_number_pattern = r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}"
    doc = nlp(string)
    extracted_phone_numbers = []
    for match in re.finditer(phone_number_pattern, doc.text):
        extracted_phone_numbers.append(match.group())
    if len(extracted_phone_numbers) != 0:
        return ','.join(extracted_phone_numbers)
    phone_number_pattern = r"\+\d{2}-\d{5}-\d{5}"
    for match in re.finditer(phone_number_pattern, doc.text):
        extracted_phone_numbers.append(match.group())
    if len(extracted_phone_numbers) != 0:
        return ','.join(extracted_phone_numbers)
    phone_number_pattern = r"\+\d{2}-\d{5} \d{5}"
    for match in re.finditer(phone_number_pattern, doc.text):
        extracted_phone_numbers.append(match.group())
    if len(extracted_phone_numbers) != 0:
        return ','.join(extracted_phone_numbers)


def get_education(path, resume_text, llm, knowledgeBase):
    """
    Extracts education details from a resume using a ResumeParser or through an OpenAI language model and knowledge base.

    :param path: Path to the resume file.
    :param resume_text: Text content of the resume.
    :param llm: OpenAI's language model instance.
    :param knowledgeBase: Knowledge base constructed from text embeddings.
    :return: Extracted education details from the resume, or None if not found.
    """
    education_new = ResumeParser(path).get_extracted_data()
    education_new = education_new['degree']
    if education_new is not None:
        return ','.join(education_new)
    if education_new is None:
        time.sleep(1)
        res = get_details_from_openai(resume_text,
                                      'what is the highest education degree give me in json format where key is degree',
                                      llm,
                                      knowledgeBase)
        if res.startswith('{'):
            res = json.loads(res)
            return res['degree']
        return None


def get_current_location(resume_text, llm, knowledgeBase):
    """
    Extracts the current location of the candidate from the resume text using OpenAI's language model and knowledge base.

    :param resume_text: Text content of the resume.
    :param llm: OpenAI's language model instance.
    :param knowledgeBase: Knowledge base constructed from text embeddings.
    :return: Current location of the candidate, or None if not found.
    """
    time.sleep(1)
    res = get_details_from_openai(resume_text,
                                  'what is the location of the candidate give me the output in json format where key is location',
                                  llm,
                                  knowledgeBase)
    if res.startswith('{'):
        res = json.loads(res)
        return res['location']
    else:
        data = ',' + ','.join(cities_data).replace('\n', '') + ','
        res = res.replace('"', '').replace(',', '').replace('.', '').split(" ")
        for w in res:
            if f',{w},'.lower() in data.lower():
                return w
    return None


def extract_name(resume_text):
    """
    Extracts the name of the candidate from the resume text.

    :param resume_text: Text content of the resume.
    :return: Extracted name of the candidate.
    """
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern], on_match=None)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        if '@' in span.text:
            return span.text.replace(get_email_addresses(resume_text), '')
        return span.text


def extract_certifications(resume_text, llm, knowledgeBase):
    """
    Extracts certifications from the resume text using OpenAI's language model and knowledge base.

    :param resume_text: Text content of the resume.
    :param llm: OpenAI's language model instance.
    :param knowledgeBase: Knowledge base constructed from text embeddings.
    :return: Extracted certifications from the resume, or None if not found.
    """
    time.sleep(1)
    r = get_details_from_openai(resume_text,
                                'what are the only certifications give me in json format where key is certifications',
                                llm,
                                knowledgeBase)
    if r.startswith("{"):
        r = json.loads(r)
        return ','.join(r['certifications'])
    return None


def get_exp(resume_text, llm, knowledgeBase):
    """
    Extracts the number of years of experience from the resume text using OpenAI's language model and knowledge base.

    :param resume_text: Text content of the resume.
    :param llm: OpenAI's language model instance.
    :param knowledgeBase: Knowledge base constructed from text embeddings.
    :return: Extracted number of years of experience from the resume, or None if not found.
    """
    time.sleep(1)
    exp = get_details_from_openai(resume_text,
                                  'what is the number of years of experience give me in json format where key is exp',
                                  llm,
                                  knowledgeBase)
    if exp.startswith("{"):
        r = json.loads(exp)
        return r['exp']
    else:
        pattern = r'(\d+(?:\.\d+)?)'
        exp = exp.replace("{", "").replace("}", "").replace('"', '')
        result1 = re.search(pattern, exp)
        exp = None
        if result1:
            exp = result1.group(1)
    return exp if len(exp) != 0 else None


def get_details(resume_text, path, llm):
    """
    Extracts various details from the resume text and file using specified methods.

    :param resume_text: Text content of the resume.
    :param path: Path to the resume file.
    :param llm: OpenAI's language model instance.
    :return: Dictionary containing extracted details from the resume.
    """
    knowledgeBase = get_knowledge_base(embeddings, resume_text)
    extracted_text = {"Name": extract_name(resume_text),
                      "E-Mail": get_email_addresses(resume_text),
                      "Phone No": get_phone_numbers(resume_text),
                      'Experience': get_exp(resume_text, llm, knowledgeBase),
                      'Education': get_education(path, resume_text, llm, knowledgeBase),
                      'Approx Current Location': get_current_location(resume_text, llm, knowledgeBase),
                      'certifications': extract_certifications(resume_text, llm, knowledgeBase),
                      'File Name': path.name
                      }
    return extracted_text


def read_pdf(file):
    """
    Reads text content from a PDF file.

    :param file: File-like object representing the PDF file.
    :return: Text content extracted from the PDF file.
    """
    save_path = Path('./', file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())
    resume_data = open(f'./{file.name}', 'rb')
    Script = PyPDF2.PdfReader(resume_data)
    pages = len(Script.pages)
    Script = []
    with pdfplumber.open(resume_data) as pdf:
        for i in range(0, pages):
            page = pdf.pages[i]
            text = page.extract_text()
            Script.append(text)
    Script = ''.join(Script)
    resume_data = Script.replace("\n", " ")
    # os.remove(save_path)
    return resume_data


def read_docx(file):
    """
    Reads text content from a DOCX file.

    :param file: File-like object representing the DOCX file.
    :return: Text content extracted from the DOCX file.
    """
    save_path = Path('./', file.name)
    with open(save_path, mode='wb') as w:
        w.write(file.getvalue())
    resume_data = open(f'./{file.name}', 'rb')
    my_text = docx2txt.process(file)
    return my_text


def preprocess_text(text):
    return re.sub(r'\W+', ' ', text).lower().split()


def jd_comparer(resume, job_description):
    resume_words = preprocess_text(resume)
    job_words = preprocess_text(job_description)
    job_word_freq = Counter(job_words)
    matched_words = Counter()
    for word in resume_words:
        if word in job_word_freq and matched_words[word] < job_word_freq[word]:
            matched_words[word] += 1
    total_matched_words = sum(matched_words.values())
    total_job_words = sum(job_word_freq.values())
    match_percentage = (total_matched_words / total_job_words) * 100
    return {"extract_name": extract_name(resume), "Phone No": get_phone_numbers(resume),
            "Score": min(match_percentage, 100)}
    pass


st.title("Welcome to Resume Chat Bot ")
uploaded_resumes = st.file_uploader(
    "Upload a resume (PDF or Docx)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)
total_files = []


# @st.experimental_singleton
def get_embeddings():
    llm = OpenAI(openai_api_key=st.secrets['api_key'], model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['api_key'])
    return embeddings, llm


embeddings, llm = get_embeddings()

if len(uploaded_resumes) != 0:
    pool = ThreadPool(min(len(uploaded_resumes), 2))
    threads = pool.map_async(
        lambda file_data: get_details(
            read_pdf(file_data) if file_data.type == 'application/pdf' else read_docx(file_data),
            file_data,
            llm
        ),
        uploaded_resumes
    )
    total_files = threads.get()
    if len(total_files) != 0:
        total_files_str = '\n'.join([f"{r}:{row[r]}" for row in total_files for r in row])
        st.write(total_files_str)
        knowledgeBase = get_knowledge_base(embeddings, total_files_str)
        question = st.text_input("Enter the query")
        if question:
            res = get_details_from_openai(total_files_str,
                                          question,
                                          llm,
                                          knowledgeBase)
            st.write(res)
        # df = pd.DataFrame(total_files)
        # df.index = np.arange(1, len(df) + 1)
        # df.index.names = ['S.No']
        # res_df = st.dataframe(df)
        # df['Phone No'] = '"' + df['Phone No'] + '"'
        # col_1, col_2 = st.columns(2)
        # col_1.download_button(
        #     "Click to Download",
        #     df.to_csv(),
        #     "file.csv",
        #     "text/csv",
        #     key='download-csv'
        # )
    pass
