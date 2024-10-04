import streamlit as st
from utils import load_data, search, get_unique_authors, get_unique_categories
import nltk
import html
import numpy as np
import gdown
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime 
from spacy.cli import download

nlp = spacy.load('en_core_web_sm')

st.set_page_config(
    page_title="Academic Paper Search Engine",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

def authenticate_google_sheets(json_keyfile):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile, scope)
    client = gspread.authorize(creds)
    return client

def log_feedback_to_sheet(client, sheet_id, feedback_data):
    try:
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.sheet1  
        worksheet.append_row(feedback_data)
        print("Feedback logged successfully.")
    except gspread.exceptions.APIError as e:
        print(f"API error occurred: {e}")
        print(f"Response content: {e.response.content}")
    except PermissionError:
        print("Permission denied: Please ensure the Google Sheet is shared with the service account.")
    except Exception as e:
        print(f"An error occurred: {e}")

file_id = '1Tjio64AEA23PZB7fT3LobVT4WKghtSK4'
data_file_path = 'academic_papers.csv'  

if not os.path.exists(data_file_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', data_file_path, quiet=False)
else:
    print(f"File {data_file_path} already exists. Skipping download.")

data, tfidf_matrix, tfidf_vectorizer = load_data(data_file_path)

if data is None:
    st.error("Failed to load data. Please check the file path and format.")
    st.stop()

authors_list = get_unique_authors(data)
categories_list = get_unique_categories(data)

if 'author_filter' not in st.session_state:
    st.session_state.author_filter = ""
if 'cat_filter' not in st.session_state:
    st.session_state.cat_filter = ""
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = None  
if 'num_relevant_docs' not in st.session_state:
    st.session_state.num_relevant_docs = 0
if 'total_docs' not in st.session_state:
    st.session_state.total_docs = 0
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

st.markdown("""<style>
    body {
        background-color: #f0f4f8;  /* Light background color */
    }
    .title {
        text-align: center;
        color: #FF6F61;  /* Coral color for the title */
        font-size: 48px; /* Larger font size */
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    .result-title {
        color: #FF8C00;  /* Dark orange for paper titles */
        font-size: 28px;
        margin-top: 15px;
    }
    .result-abstract {
        font-style: italic;
        color: #696969;  /* Dim gray for abstracts */
    }
    .divider {
        border-bottom: 2px solid #FFA07A;  /* Light salmon for dividers */
        margin: 20px 0;
    }
</style>""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Academic Paper Search Engine</h1>", unsafe_allow_html=True)

query = st.text_input("Enter your search query:", value=st.session_state.query)

with st.expander("🔍 Advanced Filtering Options", expanded=False):
    if st.session_state.author_filter in authors_list:
        author_filter_index = authors_list.index(st.session_state.author_filter) + 1
    else:
        author_filter_index = 0

    author_filter = st.selectbox(
        "Filter by Author (optional):",
        options=[""] + authors_list, 
        index=author_filter_index
    )

    if st.session_state.cat_filter in categories_list:
        cat_filter_index = categories_list.index(st.session_state.cat_filter) + 1
    else:
        cat_filter_index = 0

    cat_filter = st.selectbox(
        "Filter by Categories (optional):",
        options=[""] + categories_list,  
        index=cat_filter_index
    )

    st.session_state.author_filter = author_filter
    st.session_state.cat_filter = cat_filter

    if st.button("Clear Filters"):
        st.session_state.author_filter = ''
        st.session_state.cat_filter = ''
        st.write("Filters and search query have been cleared.")

if st.button("Search", key="search_button"):
    if query:
        
        st.session_state.query = query
        
        with st.spinner("Searching for papers... 🚀"):
            results = search(data, query, tfidf_matrix, tfidf_vectorizer, author_filter, cat_filter)

        results = results[results['final_similarity'].notnull() & (results['final_similarity'] > 0.2)]

        st.session_state.search_results = results
        st.session_state.total_docs = len(results)

if st.session_state.search_results is not None and not st.session_state.search_results.empty:
    st.write("**Search Results:**")
    for index, row in st.session_state.search_results.iterrows():
        st.markdown(f"<h2 class='result-title'>{row['title']}</h2>", unsafe_allow_html=True)
        st.write(f"**Authors:** {row['authors_extracted']}")
        st.markdown(f"<p class='result-abstract'>{html.escape(row['abstract'])}</p>", unsafe_allow_html=True)
        st.write(f"**Categories:** {row['categories']}")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


if st.session_state.search_results is not None:
    st.subheader("Feedback on Relevance")

    
    st.write(f"Total Documents: {st.session_state.total_docs}")

    if st.session_state.total_docs > 0:
        num_relevant_docs = st.number_input(
            "How many documents are relevant?", 
            min_value=0,
            max_value=st.session_state.total_docs, 
            value=st.session_state.num_relevant_docs
        )
    else:
        st.warning("No documents available to evaluate. Please refine your search.")
        num_relevant_docs = 0 

    
    st.session_state.num_relevant_docs = num_relevant_docs

    
    json_keyfile = 'C:/Users/Chathuka/Desktop/Y3S1/proj/irwa/acadamic-search-engine-3dc54b7224d2.json' 
    client = authenticate_google_sheets(json_keyfile)

   
    if st.button("Submit Feedback") and st.session_state.total_docs >= 1:
        st.session_state.feedback_submitted = True
        st.success("Feedback submitted successfully!")

        precision = st.session_state.num_relevant_docs / st.session_state.total_docs if st.session_state.total_docs > 0 else 0
        recall = st.session_state.num_relevant_docs / st.session_state.total_docs if st.session_state.total_docs > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0


        ground_truth_relevant_docs = st.session_state.total_docs
        mae = abs(ground_truth_relevant_docs - st.session_state.num_relevant_docs)
        rmse = np.sqrt((ground_truth_relevant_docs - st.session_state.num_relevant_docs) ** 2)

       
        feedback_data = [
            st.session_state.query,  
            st.session_state.num_relevant_docs,
            st.session_state.total_docs,
            precision,
            recall,
            f1_score,
            mae,
            rmse,
            st.session_state.author_filter,  
            st.session_state.cat_filter,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        print("Feedback Data to be logged:", feedback_data)

        log_feedback_to_sheet(client, "1MqYacRMIGTjsMW3AX27YYyzyGkK1Aydqe3aieC2Nw-E", feedback_data)

        # # Display the metrics
        # st.write("### Metrics:")
        # st.write(f"- Precision: {precision:.2f}")
        # st.write(f"- Recall: {recall:.2f}")
        # st.write(f"- F1-score: {f1_score:.2f}")
        # st.write(f"- Mean Absolute Error (MAE): {mae:.2f}")
        # st.write(f"- Root Mean Squared Error (RMSE): {rmse:.2f}")

        # # Additional debug printouts
        # st.write(f"Total Documents: {st.session_state.total_docs}")
        # st.write(f"Number of Relevant Documents: {st.session_state.num_relevant_docs}")
