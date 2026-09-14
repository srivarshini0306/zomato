import os
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector

from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "openai/gpt-oss-120b"

NEW_REVIEWS = 500
TOP_K = 5

CACHE_FILE = "review_embeddings.parquet"


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ============================================================
# READ REVIEWS FROM SNOWFLAKE
# ============================================================

def read_reviews_from_snowflake():

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )

    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({NEW_REVIEWS} ROWS)
    """

    df = conn.cursor().execute(query).fetch_pandas_all()

    conn.close()

    df.columns = [col.lower() for col in df.columns]

    return df


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def embed(texts):

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


# ============================================================
# LOAD REVIEWS + EMBEDDINGS
# ============================================================

@st.cache_data
def load_reviews():

    if os.path.exists(CACHE_FILE):

        return pd.read_parquet(CACHE_FILE)

    df = read_reviews_from_snowflake()

    df["embedding"] = embed(
        df["comment"].fillna("").tolist()
    )

    df.to_parquet(CACHE_FILE)

    return df


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Chat with your Zomato Reviews")

st.caption(
    f"Searching {NEW_REVIEWS} reviews "
    f"using {EMBEDDING_MODEL} embeddings "
    f"and answering with {CHAT_MODEL}"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vec_a, vec_b):

    return np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) *
        np.linalg.norm(vec_b)
    )


# ============================================================
# FIND SIMILAR REVIEWS
# ============================================================

def find_similar_reviews(question, df):

    question_vector = embed([question])[0]

    scores = []

    for review_vector in df["embedding"]:

        scores.append(
            cosine_similarity(
                question_vector,
                review_vector
            )
        )

    df = df.copy()

    df["score"] = scores

    return df.nlargest(
        TOP_K,
        "score"
    )


# ============================================================
# ASK GROQ
# ============================================================

def ask_llm(question, top_reviews):

    context = ""

    for _, row in top_reviews.iterrows():

        context += (
            f"City: {row['city']}\n"
            f"Rating: {row['rating']} stars\n"
            f"Review: {row['comment']}\n\n"
        )


    system_prompt = """
You are a helpful assistant analyzing Zomato customer reviews.

Answer ONLY using the customer reviews provided in the context.

Do not invent information.

If the provided reviews do not contain enough information
to answer the question, say that the available reviews
do not provide enough information.

Be concise and clear.
"""


    user_prompt = f"""
Question:
{question}

Customer Reviews:
{context}
"""


    response = client.chat.completions.create(

        model=CHAT_MODEL,

        temperature=0.2,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# LOAD DATA
# ============================================================

review_df = load_reviews()


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder=(
        "e.g. What are the most common "
        "complaints about delivery?"
    )
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    top_reviews = find_similar_reviews(
        question,
        review_df
    )

    answer = ask_llm(
        question,
        top_reviews
    )

    st.markdown("**Answer:**")

    st.write(answer)


    with st.expander(
        "Reviews used to build this answer"
    ):

        st.dataframe(
            top_reviews[
                [
                    "city",
                    "rating",
                    "comment"
                ]
            ],
            hide_index=True
        )