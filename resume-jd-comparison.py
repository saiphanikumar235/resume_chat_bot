import os

import numpy as np
import nltk

nltk.download('stopwords')
import pandas as pd
import numpy as np
import PyPDF2, pdfplumber, nlp, re, docx2txt, streamlit as st, nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from nltk.corpus import stopwords
from pathlib import Path
import json
from pyresparser import ResumeParser
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time

# os.system("python -m spacy download en_core_web_sm")
# os.system("python -m nltk.downloader words")
# os.system("python -m nltk.downloader stopwords")
nltk.download('punkt')

knowledgeBase = None


def get_knowledge_base(embeddings, text):
    api_key = st.secrets['api_key']
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    # embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    global knowledgeBase
    knowledgeBase = FAISS.from_texts(chunks, embeddings)


def get_details_from_openai(text, query, llm):
    api_key = st.secrets['api_key']
    docs = knowledgeBase.similarity_search(query)
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=query)
    return response


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def compare_jd(resume_text, jd):
    if jd != '':
        resume_tokens = word_tokenize(resume_text.lower())
        job_desc_tokens = word_tokenize(jd.lower())
        model = Word2Vec([resume_tokens, job_desc_tokens], vector_size=100, window=5, min_count=1, sg=0)
        resume_vector = np.mean([model.wv[token] for token in resume_tokens], axis=0)
        job_desc_vector = np.mean([model.wv[token] for token in job_desc_tokens], axis=0)
        MatchPercentage = cosine_similarity(resume_vector, job_desc_vector) * 100
        # Req_Clear = ''.join(open("./req.txt", 'r', encoding="utf8").readlines()).replace("\n", "")
        # jd_text = jd
        # Match_Test = [resume_text.lower(), jd_text.lower()]
        # cv = TfidfVectorizer()
        # count_matrix = cv.fit_transform(Match_Test)
        # MatchPercentage = cosine_similarity(count_matrix[0], count_matrix[1])
        # MatchPercentage = round(MatchPercentage[0][0]*100, 2)
        # print('Match Percentage is :' + str(MatchPercentage) + '% to Requirement')
        return MatchPercentage
    return "No JD to Compare"


def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return ','.join(list(set(r.findall(string))))


def get_phone_numbers(string):
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

    # r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    # phone_numbers = r.findall(string)
    # return ','.join(list(set([re.sub(r'\D', '', num) for num in phone_numbers])))


def get_education(path, resume_text, llm):
    education_new = ResumeParser(path).get_extracted_data()
    education_new = education_new['degree']
    if education_new is None:
        res = get_details_from_openai(resume_text,
                                      'what is the highest education degree give me in json format where key is degree',
                                      llm)
        st.write(res)
        if res.startswith('{'):
            res = json.loads(res)
            # time.sleep(60)
            return res['degree']
        return None
    else:
        return ','.join(education_new)


def get_current_location(resume_text, llm):
    res = get_details_from_openai(resume_text,
                                  'what is the location of candiate give me in json format where key is location',
                                  llm)
    st.write(res)
    if res.startswith('{'):
        res = json.loads(res)
        # st.write(res)
        # time.sleep(60)
        return res['location']
    return None


def extract_name(resume_text):
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern], on_match=None)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        if span.text.lower() in [r.lower().replace("\n", "") for r in
                                 open('./linkedin skill', 'r', encoding="utf8").readlines()]:
            return re.sub(r'\d', '', get_email_addresses(resume_text).split('@')[0]).capitalize()
        if '@' in span.text:
            return span.text.replace(get_email_addresses(resume_text), '')
        return span.text


def get_skills(resume_text):
    nlp = spacy.load('en_core_web_sm')
    nlp_text = nlp(resume_text)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = [r.lower().replace("\n", "") for r in open('./linkedin skill', 'r', encoding="utf8").readlines()]
    skillset = []
    for i in tokens:
        if i.lower() in skills:
            skillset.append(i)
    for i in nlp_text.noun_chunks:
        i = i.text.lower().strip()
        if i in skills:
            skillset.append(i)

    return ','.join([word.capitalize() for word in set([word.lower() for word in skillset])])


def extract_certifications(resume_text, llm):
    r = get_details_from_openai(resume_text,
                                'what are the only certifications give me in json format where key is certifications',
                                llm)
    st.write(r)
    if r.startswith("{"):
        r = json.loads(r)
        return ','.join(r['certifications'])
    return None


def get_exp(resume_text, llm):
    words_to_numbers = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'zero': '0'
    }
    pattern = re.compile(r'\b(' + '|'.join(words_to_numbers.keys()) + r')\b')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "DATE" and ent.text.lower() in ["year", 'yrs', 'years']:
            years_of_experience = ent.text
            for y in years_of_experience.split():
                if '.' in y:
                    return y
                if y.lower() in words_to_numbers.keys() or y.replace('+', '').isnumeric():
                    years = f"{y.replace('+', '')}+"
                    return re.sub(pattern, lambda x: words_to_numbers[x.group()], years)
    # for ent in doc.ents:
    #     if ent.label_ == 'CARDINAL':
    #         years_of_experience = ent.text
    #         for y in years_of_experience.split():
    #             if '.' in y and '+' in y:
    #                 return y.replace('++', '+')
    #             if y.lower() in words_to_numbers.keys() or y.isnumeric():
    #                 print(y)
    #                 years = f'{y}+'
    #                 return re.sub(pattern, lambda x: words_to_numbers[x.group()], years)
    exp = get_details_from_openai(resume_text,
                                  'Find the number of years of experience and give me output in json format where key is exp',
                                  llm)
    st.write(exp)
    if exp.startswith("{"):
        r = json.loads(exp)
        return ','.join(r['exp'])
    return ','.join(exp) if len(exp) != 0 else None


def get_details(resume_text, path, llm):
    extracted_text = {"Name": extract_name(resume_text),
                      "E-Mail": get_email_addresses(resume_text),
                      "Phone No": get_phone_numbers(resume_text),
                      'Skills': get_skills(resume_text),
                      'Experience': get_exp(resume_text, llm),
                      'Education': get_education(path, resume_text, llm),
                      'Approx Current Location': get_current_location(resume_text, llm),
                      'certifications': extract_certifications(resume_text, llm)
                      }
    return extracted_text


def read_pdf(file):
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
    return resume_data


def read_docx(file):
    my_text = docx2txt.process(file)
    return my_text


st.title("Rapid-Recruit-X")
# jd = st.text_input('please enter the job description below:')
uploaded_resumes = st.file_uploader(
    "Upload a resume (PDF or Docx)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
total_files = []


@st.experimental_singleton
def get_embeddings():
    llm = OpenAI(openai_api_key=st.secrets['api_key'], model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['api_key'])
    return embeddings, llm


embeddings, llm = get_embeddings()

for index, uploaded_resume in enumerate(uploaded_resumes):
    if uploaded_resume.type == "application/pdf":
        resume_text = read_pdf(uploaded_resume)
    else:
        resume_text = read_docx(uploaded_resume)
    get_knowledge_base(embeddings, resume_text)
    resume_details = get_details(resume_text, uploaded_resume, llm)
    # resume_details['Resume Score'] = compare_jd(resume_text, jd)
    resume_details['File Name'] = uploaded_resume.name
    total_files.append(
        resume_details
    )
    time.sleep(10)
if len(total_files) != 0:
    df = pd.DataFrame(total_files)
    df.index = np.arange(1, len(df) + 1)
    df.index.names = ['S.No']
    res_df = st.dataframe(df)
    df['Phone No'] = '"' + df['Phone No'] + '"'
    st.download_button(
        "Click to Download",
        df.to_csv(),
        "file.csv",
        "text/csv",
        key='download-csv'
    )
