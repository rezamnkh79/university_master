import os
import time
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import lancedb
import pyarrow as pa
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import LanceDB
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from IPython.display import Image, display
from typing import Literal
import json
import re
import numpy as np
from db_manager import cancel_order, check_order_status, comment_order
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from db_manager import food_search
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


class IsRelated(BaseModel):
    reasoning: str = Field(
        description="The reasoning behind the decision, whether the query is related to baking and cooking or not.")
    is_related_flag: bool = Field(description="The decision whether the query is related to baking and cooking or not.")


AVALAI_BASE_URL = "https://api.avalai.ir/v1"
GPT_MODEL_NAME = "gpt-4o-mini"
TAVILY_API_KEY = "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"
LANCE_DB_PATH = "food_knowledge.lance"
SIMILARITY_THRESHOLD = 0.75
Question_THRESHOLD = 0.65

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
gpt4o_chat = ChatOpenAI(
    model=GPT_MODEL_NAME,
    base_url=AVALAI_BASE_URL,
    api_key="aa-yziC62HWTHrm7jBNtaPt0Ph3jSFRtgyQAfYGD6wetWbfCkOO",
)

LLAMA_CLOUD_API_KEY = "llx-a2y9jRnOuKNRohPrkyjCSylHDItkFq1egvDYIjeXGGYEdjnH"

LOG_FILE = "query_log.txt"
PDF_FILE_PATH = "The_New_Complete_Book_of_Food.pdf"
LANCE_DB_PATH = "food_knowledge.lance"
DB_PATH = "food_orders.db"


# =================================================================================================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks


def parse_pdf():
    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY)
    file_extractor = {".pdf": parser}

    data_for_parse = SimpleDirectoryReader(input_files=[PDF_FILE_PATH], file_extractor=file_extractor)
    documents = data_for_parse.load_data()

    return documents


def df_to_dict_batches(df, batch_size: int = 128):
    for start_idx in range(0, len(df), batch_size):
        end_idx = start_idx + batch_size
        batch_dicts = df.iloc[start_idx:end_idx].to_dict(orient="records")
        yield batch_dicts


embedding_model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")


class ChunksOfData(LanceModel):
    id: str
    text: str = embedding_model.SourceField()
    metadata_file_name: str
    metadata_creation_date: str
    metadata_pagenumber: int
    vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()


def store_embeddings(df):
    db_connection = lancedb.connect("./food_knowledge.lance")

    db_connection.create_table(
        "embedded_chunks3",
        data=df_to_dict_batches(df, batch_size=10),
        schema=ChunksOfData,
    )
    return db_connection


def process_store_book_in_db():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )
    documents_list = []
    page_number = 0
    documents = parse_pdf()
    for doc in documents:

        texts = text_splitter.split_text(doc.text)
        for text in texts:
            item = dict()
            item["id"] = str(doc.id_)
            item["text"] = text
            item["metadata_file_name"] = doc.metadata["file_name"]
            item["metadata_creation_date"] = doc.metadata["creation_date"]
            item["metadata_pagenumber"] = page_number
            documents_list.append(item)
    chunks = split_text(documents_list)
    df = pd.DataFrame(chunks)
    store_embeddings(df)


# =================================================================================================================
def log_request(user_input, response):
    """
    Logs the user query and system response to a file with a timestamp.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("============================================================================================\n")
        file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"User Request: {user_input}\n")
        file.write(f"System Response: {response}\n")
        file.write("==============================\n")


def log_recommend(user_refined_list, recommendations_list):
    """
    Logs the user query and system response to a file with a timestamp.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("============================================================================================\n")
        file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"User Refined List: {user_refined_list}\n")
        file.write(f"System Recommends List: {recommendations_list}\n")
        file.write("==============================\n")


# =======================================================================================================================
# query about food
class QueryState:
    def __init__(self, query):
        self.query = query
        self.context = ""
        self.db_results = []
        self.online_results = []
        self.final_response = ""
        self.graph_state = {}

    def update_context(self, results):
        self.context = "\n".join(results)

    def set_final_response(self, response):
        self.final_response = response

    def set_graph_state(self, graph_state):
        self.graph_state = graph_state


def log_to_file(state):
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write("============================================================================================\n")
        file.write(f"Query: {state.query}\n")
        file.write(f"Database Results: {state.db_results}\n")
        file.write(f"Online Results: {state.online_results}\n")
        file.write(f"Graph State: {state.graph_state}\n")
        file.write(f"Final Response: {state.final_response}\n")
        file.write(f"====================\n\n")


def search_in_db(query):
    db_connection = lancedb.connect(LANCE_DB_PATH)
    table = db_connection.open_table("embedded_chunks3")
    query_embedding = embeddings.embed_query(query)
    results = table.search(query_embedding).limit(20).to_pandas()

    matches = [row["text"] for _, row in results.iterrows() if
               np.dot(query_embedding, row["vector"]) >= SIMILARITY_THRESHOLD]
    return matches


def search_online(query):
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    params = {"query": query, "num_results": 3}
    response = requests.post(url, headers=headers, json=params)

    if response.status_code == 200:
        return [row["content"] for row in response.json()["results"]]
    return []


def decide_retrieval(state):
    state.db_results = search_in_db(state.query)
    if len(state.db_results) == 0:
        state.online_results = search_online(state.query)
    if state.db_results:
        return "database", state
    elif state.online_results:
        return "online", state
    else:
        return "none", state


def retrieve_from_db(state):
    state.update_context(state.db_results)
    return state


def retrieve_from_online(state):
    state.update_context(state.online_results)
    return state


def retrieve_both(state):
    state.update_context(state.db_results + state.online_results)
    return state


def generate_response(state):
    if not state.context:
        return state.set_final_response("I'm sorry, I don't have any information on this topic.")
    time.sleep(7)
    messages = [
        SystemMessage(
            "Answer user query based on the given context.if the context is not related please give a food message for user that we can't answer to this question"),
        HumanMessage(f"Question:\n{state.query}\nContext:\n{state.context}")
    ]
    response = gpt4o_chat.invoke(messages)
    state.set_final_response(response.content)
    return state


def is_related(state):
    query = state.query

    llm_with_structured_output = gpt4o_chat.with_structured_output(IsRelated)

    response = llm_with_structured_output.invoke(query)

    print(response)

    if response.is_related_flag:
        return "node_query_rewrite"
    else:
        return "node_generate_answer"


def build_graph(state):
    builder = StateGraph(QueryState)
    builder.add_node("node_query_rewrite", lambda state: state)
    builder.add_node("node_search_internet", lambda state: state)
    builder.add_node("node_generate_answer", lambda state: generate_response(state))

    builder.add_conditional_edges(START, is_related)

    if state.db_results and state.online_results:
        builder.add_edge("node_query_rewrite", "node_search_internet")
        builder.add_edge("node_search_internet", "node_generate_answer")
    elif state.db_results:
        builder.add_edge("node_query_rewrite", "node_generate_answer")
    elif state.online_results:
        builder.add_edge("node_query_rewrite", "node_search_internet")
        builder.add_edge("node_search_internet", "node_generate_answer")
    else:
        builder.add_edge("node_query_rewrite", "node_generate_answer")

    builder.add_edge("node_generate_answer", END)

    graph_image_name = f"graph_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    graph_image_path = os.path.join("graph_image", graph_image_name)
    os.makedirs("graph_image", exist_ok=True)

    graph_image = builder.compile().get_graph().draw_mermaid_png()
    with open(graph_image_path, "wb") as image_file:
        image_file.write(graph_image)

    return graph_image_path


def process_query(query):
    state = QueryState(query)

    retrieval_type, state = decide_retrieval(state)

    if retrieval_type == "database":
        state = retrieve_from_db(state)
    elif retrieval_type == "online":
        state = retrieve_from_online(state)
    elif retrieval_type == "both":
        state = retrieve_both(state)
    state = generate_response(state)
    log_to_file(state)
    build_graph(state)
    return state.final_response


def validate_question_scope(query):
    food_keywords = [
        "food", "baking", "cooking", "ingredients", "recipes", "restaurant", "menu", "fruit", "vegetable", "nutrients",
        "meal"
    ]
    food_keywords_embeddings = embeddings.embed_documents(food_keywords)
    query_embedding = embeddings.embed_query(query)
    similarity_score = np.dot(food_keywords_embeddings, query_embedding)
    if max(similarity_score) < Question_THRESHOLD:
        return "This question is outside the chatbot's domain."
    return None


def answer_food_question(query):
    validation_msg = validate_question_scope(query)
    if validation_msg:
        return validation_msg
    return process_query(query)


# ===================================================================================================================================
# cancel status comment
def extract_slot_values(user_input, required_slots):
    """
    Use GPT-4o to extract relevant slot values from user input.
    """
    time.sleep(7)
    messages = [
        {"role": "system",
         "content": "Extract the required slot values for the user intent. The slots are: order_id, phone_number, person_name, comment. Identify the missing fields and return them as a JSON object."},
        {"role": "user", "content": user_input}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def request_missing_info(session_data):
    """
    Request missing information from the user based on the specific intent type.
    """
    required_fields = {
        "cancel": ["order_id", "phone_number"],
        "status": ["order_id"],
        "comment": ["order_id", "person_name", "comment"]
    }

    while True:
        missing_info = [slot for slot in required_fields[session_data["intent"]] if
                        session_data["required_slots"].get(slot) is None]
        if not missing_info:
            break

        user_input = input(
            f"For {session_data['intent']} request, please provide: {', '.join(missing_info)}.\nEnter details: ")
        extracted_data = extract_slot_values(user_input, session_data["required_slots"])
        extracted_data = json.loads(
            extracted_data.strip().replace("json", "", 1).replace("\n", "").replace("`", "").strip())

        for slot, value in extracted_data.items():
            if value:
                session_data["required_slots"][slot] = value

    return session_data["required_slots"]


def process_customer_request(session_data):
    """
    Step 1: Ask the user for their intent.
    Step 2: Once intent is detected, request required details.
    Step 3: Validate and process the request only when all fields are filled.
    """
    intent_templates = {
        "cancel": "I want to cancel my order",
        "status": "I want to check my order status",
        "comment": "I want to leave a comment on my order"
    }
    intent_embeddings = {intent: embeddings.embed_query(template) for intent, template in intent_templates.items()}
    if "intent" not in session_data:
        user_intent = input("What would you like to do? (Cancel Order / Check Order Status / Leave a Comment): ")

        user_intent_embedding = embeddings.embed_query(user_intent)
        similarities = {intent: np.dot(user_intent_embedding, template_embedding)
                        for intent, template_embedding in intent_embeddings.items()}
        session_data["intent"] = max(similarities, key=similarities.get)

        required_slots = {
            "cancel": {"order_id": None, "phone_number": None},
            "status": {"order_id": None},
            "comment": {"order_id": None, "person_name": None, "comment": None}
        }[session_data["intent"]]

        session_data["required_slots"] = required_slots

    missing_info = [slot for slot, value in session_data["required_slots"].items() if value is None]
    if missing_info:
        user_input = input(f"Please provide the following details: {', '.join(missing_info)}.\nEnter details: ")
        extracted_data = extract_slot_values(user_input, session_data["required_slots"])
        extracted_data = json.loads(
            extracted_data.strip().replace("json", "", 1).replace("\n", "").replace("`", "").strip())

        for slot, value in extracted_data.items():
            session_data["required_slots"][slot] = value

    session_data["required_slots"] = request_missing_info(session_data)

    if session_data["intent"] == "cancel":
        result = cancel_order(int(session_data["required_slots"]["order_id"]),
                              session_data["required_slots"]["phone_number"])
        session_data.clear()
        return result
    elif session_data["intent"] == "status":
        result = check_order_status(int(session_data["required_slots"]["order_id"]))
        session_data.clear()
        return result
    elif session_data["intent"] == "comment":
        result = comment_order(int(session_data["required_slots"]["order_id"]),
                               session_data["required_slots"]["person_name"], session_data["required_slots"]["comment"])
        session_data.clear()
        return result

    session_data.clear()
    return "I'm sorry, I didn't understand your request. Please try again."


# =======================================================================================================================================
# food search with name or restaurant


def extract_food_search_values(user_input):
    """
    Use GPT-4o to extract food name and restaurant name from user input.
    """
    time.sleep(7)
    messages = [
        {"role": "system",
         "content": "Extract the food name and restaurant name from the user's input. Return the results as a JSON object with fields: food_name and restaurant_name."},
        {"role": "user", "content": user_input}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def format_food_results(results):
    """
    Use GPT-4o to generate a natural language response for the food search results.
    """
    time.sleep(7)
    messages = [
        {"role": "system",
         "content": "Format the given food search results into a natural language response for the user."},
        {"role": "user", "content": f"Here are the matching food items: {results}"}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def request_missing_food_info(session_data):
    """
    Request missing information from the user based on required fields.
    """
    required_fields = ["food_name", "restaurant_name"]

    while True:
        missing_info = [slot for slot in required_fields if session_data["search_criteria"].get(slot) is None]
        if len(missing_info) == 2:
            user_input = input(f"To search for food, please provide: {', '.join(missing_info)}.\nEnter details: ")
            extracted_data = extract_food_search_values(user_input)
            extracted_data = json.loads(
                extracted_data.strip().replace("json", "", 1).replace("\n", "").replace("`", "").strip())

            for slot, value in extracted_data.items():
                if value:
                    session_data["search_criteria"][slot] = value
        else:
            break

    return session_data["search_criteria"]


def process_food_search(session_data):
    """
    Step 1: Ask the user for food search input.
    Step 2: Extract required details and request missing info if needed.
    Step 3: Perform food search in the database and return results.
    """
    if "search_criteria" not in session_data:
        session_data["search_criteria"] = {"food_name": None, "restaurant_name": None}

    user_input = input("What food are you looking for? You can also specify a restaurant: ")
    extracted_data = extract_food_search_values(user_input)
    extracted_data = json.loads(
        extracted_data.strip().replace("json", "", 1).replace("\n", "").replace("`", "").strip())

    for slot, value in extracted_data.items():
        session_data["search_criteria"][slot] = value

    session_data["search_criteria"] = request_missing_food_info(session_data)

    results = food_search(food_name=session_data["search_criteria"]["food_name"],
                          restaurant_name=session_data["search_criteria"]["restaurant_name"])

    if results:
        return format_food_results(results)
    else:
        return "Sorry, no matching food items were found. try again"
    session_data.clear()


# =======================================================================================================================================
# food suggestion


def analyze_food_preferences(user_input):
    """
    Use a multi-step reasoning approach (Plan and Execute, Reflexion, ReAct) to infer food types from user input.
    """
    time.sleep(8)
    messages = [
        {"role": "system",
         "content": "Analyze the user's input using a multi-step reasoning approach (Plan and Execute, Reflexion, ReAct) to infer up to 10 food types that match their preferences.just give a list of food name without detail"},
        {"role": "user", "content": user_input}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def refine_food_suggestions(initial_request, previous_suggestions, user_feedback):
    """
    Use a multi-step reasoning approach (Plan and Execute, Reflexion, ReAct) to refine food suggestions
    based on the user's initial request, their latest request, previous suggestions, and their feedback.
    """
    time.sleep(7)
    messages = [
        {"role": "system",
         "content": (
             "Analyze the user's initial request, latest input, the previous food suggestions, and their feedback on those suggestions. "
             "Use a multi-step reasoning approach (Plan and Execute, Reflexion, ReAct) to generate a refined list of up to 10 new food suggestions. "
             "Just provide a list of food names without additional details.It is so important that don't say foods that is in Previous suggestions "
         )},
        {"role": "user", "content": f"User's initial request: {initial_request}"},
        {"role": "user", "content": f"Previous suggestions given as list: {previous_suggestions}"},
        {"role": "user", "content": f"User's feedback on previous suggestions as list: {user_feedback}"}
    ]

    response = gpt4o_chat.invoke(messages)
    return response.content


def extract_food_search_values(user_input):
    """
    Use GPT-4o to extract food name and restaurant name from user input.
    """
    time.sleep(7)
    messages = [
        {"role": "system",
         "content": "Extract the food name and restaurant name from the user's input. Return the results as a JSON object with fields: food_name and restaurant_name."},
        {"role": "user", "content": user_input}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def format_food_recommendations(food_items):
    """
    Use GPT-4o to generate a natural language response for food recommendations.
    """
    time.sleep(10)
    messages = [
        {"role": "system",
         "content": "Format the given food recommendations into a natural and engaging response with best and beautiful format for the user."},
        {"role": "user",
         "content": f"Based on your preferences, here are some food recommendations. select best 5 foods that you think is good: {food_items}"}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def get_feedback_and_refine(user_feedback):
    """
    Ask GPT-4o to generate clarifying questions based on the user's feedback.
    """
    time.sleep(10)
    messages = [
        {"role": "system",
         "content": "Generate clarifying questions based on the user's feedback to better understand their food preference.ask just one best question for this situation"},
        {"role": "user", "content": user_feedback}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content


def process_food_recommendation(first_user_input):
    """
    Step 1: Ask the user for food preferences.
    Step 2: Use a multi-step reasoning approach to infer possible food categories.
    Step 3: Search for matching food items in the database.
    Step 4: Provide a natural recommendation response.
    Step 5: Get feedback from the user and refine recommendations if necessary.
    """
    food_types = analyze_food_preferences(first_user_input)
    food_list = [re.sub(r'^\d+\.\s*', '', line) for line in food_types.splitlines()]

    all_results = []
    for food_name in food_list:
        results = food_search(food_name=food_name)
        if results:
            all_results.extend(results)
    recommendations_list = []
    user_refined_list = []
    if all_results:
        recommendations = format_food_recommendations(all_results)
        print(recommendations)
        recommendations_list.append(recommendations)
        while True:
            user_feedback = input("Do these recommendations work for you?")
            user_feedback_embedding = embeddings.embed_query(user_feedback)

            negative_feedback_templates = ["no", "not really", "something else", "I don't like these",
                                           "not what I want",
                                           "it is so bad", "bad", "I don't like this foods"]
            negative_feedback_embeddings = [embeddings.embed_query(template) for template in
                                            negative_feedback_templates]

            similarity_scores = [np.dot(user_feedback_embedding, template_embedding) for template_embedding in
                                 negative_feedback_embeddings]

            if max(similarity_scores) > 0.75:
                clarifying_questions = get_feedback_and_refine(user_feedback)
                print(clarifying_questions)

                refined_input = input("Please provide more details based on these questions: ")
                user_refined_list.append(refined_input)
                refined_food_types = refine_food_suggestions(initial_request=first_user_input,
                                                             previous_suggestions=recommendations_list,
                                                             user_feedback=user_refined_list)
                refined_food_list = [re.sub(r'^\d+\.\s*', '', line) for line in refined_food_types.splitlines()]
                refined_results = []
                for food_name in refined_food_list:
                    results = food_search(food_name=food_name)
                    if results:
                        refined_results.extend(results)

                if refined_results:
                    new_recommendation = format_food_recommendations(refined_results)
                    print(new_recommendation)
                    recommendations_list.append(new_recommendation)
                else:
                    return "I still couldn't find an exact match, but maybe you can try something new!"
            else:
                log_recommend(user_refined_list, recommendations_list)
                return "I hope you enjoy eating these foods"

    else:
        return "I couldn't find exact matches, but let me suggest something similar!"


def classify_user_intent(user_input):
    """
    Uses GPT-4o to classify the user's intent into predefined categories.
    """
    time.sleep(10)
    messages = [
        {"role": "system", "content": "Classify the user's request into one of the following categories: "
                                      "cancel_order, check_status, leave_comment, food_search, "
                                      "food_recommendation, general_food_inquiry. "
                                      "If the user expresses hunger (e.g., 'I'm hungry', 'I want food', 'I'm starving') "
                                      "or thirst (e.g., 'I'm thirsty', 'I need a drink'), classify it as 'food_recommendation'. "
                                      "Return only the category name."},
        {"role": "user", "content": user_input}
    ]
    response = gpt4o_chat.invoke(messages)
    return response.content.strip().lower()


def process_user_request():
    while True:
        user_input = input("How can I assist you? ")

        intent = classify_user_intent(user_input)

        if intent == "cancel_order" or intent == "check_status" or intent == "leave_comment":
            session_data = {}
            response = process_customer_request(session_data)

        elif intent == "food_search":
            session_data = {}
            response = process_food_search(session_data)


        elif intent == "food_recommendation":
            response = process_food_recommendation(user_input)

        elif intent == "general_food_inquiry":
            response = answer_food_question(user_input)

        else:
            response = "I'm not sure how to handle that request. Could you clarify?"

        print(response)
        log_request(user_input, response)


if __name__ == "__main__":
    # process_store_book_in_db()
    process_user_request()
