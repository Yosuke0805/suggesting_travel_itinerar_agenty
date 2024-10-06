''' coding: utf-8 '''
# ------------------------------------------------------------
# Content : AI agent to suggest travel itinerary
# Author : Yosuke Kawazoe
# Data Updated：
# Update Details：
# ------------------------------------------------------------

# Import
import os
import traceback
import tempfile
import logging

import streamlit as st
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.tracers import LangChainTracer
from langgraph.prebuilt import create_react_agent

# config
# API
load_dotenv()
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if LANGCHAIN_API_KEY is None:
    LANGCHAIN_API_KEY = st.secrets["api_keys"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY is None:
    TAVILY_API_KEY = st.secrets["api_keys"]["TAVILY_API_KEY"]
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Define the system message for the agent's prompt.
system_message = """
            You are a professional travel assistant that helps people plan trips especially in Europe.
            Your task is to provide a detailed itinerary for the trip based on prerequisites.
            Please exclude countries and cities that I have already visited when you plan the itinerary.

            ### prerequisites
            - duration : {traveling_days} days
            - departure : {departure}
            - final destination : {final_destination}

            ### countries and cities that already visited
            - UK
            - Sweden
            - Finland
            - Switzerland
            - Czechia prague
            - Germany Munich
            - Italy Milan, Rome
            - Spain Madrid
            - Hungary Budapest
            - Croatia Dubrovnik
            - North Macedonia
            - Serbia Belgrade
            - Bulgaria Sophia
            - Austria Vienna
            - Slovakia Bratislava
            - Motenegro
            """
# set up tracer
tracer = LangChainTracer()

# set up tool
search = TavilySearchResults(max_results=2)
tools = [search]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# ★★★★★★  main part ★★★★★★
# ------------------------------------------------------------
def main():

    try:
        # Streamlit app
        st.title("🌍 Travel Chatbot - Let's Plan Your Trip")

        # Sidebar selection
        user_type = st.sidebar.radio("Who is using this?", ("Me", "Others"))

        if user_type == "Me":
            # for local environment: Load environment variables from .env file
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            # for Streamlit Community Cloud : load API key using Streamlit secrets
            if GEMINI_API_KEY is None:
                # Login section
                password = st.sidebar.text_input("Password", type="password")
                stored_password = st.secrets["password"]["MY_PASSWORD"]
                if password:
                    if password == stored_password:
                        st.sidebar.success("Welcome Yosuke!")
                        GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]
                        # configure model with api key
                        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
                    else:
                        st.sidebar.error("Invalid password")
            else:
                os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        elif user_type == "Others":
            # set Gemini API
            GEMINI_API_KEY = st.sidebar.text_input("Input your Gemini API key", type="password")
            # configure model with api key
            os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

        # configure model
        model = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )

        try:
            # Initialize session state for message_history and conversation history
            if "message_history" not in st.session_state:
                st.session_state["message_history"] = []
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            # show previous conversations
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Get the user's query input
            query = st.chat_input("What's up?")

            # If user submits a query
            if query:
                st.session_state["messages"].append({"role": "user", "content": query})

                with st.chat_message("user"):
                    st.markdown(query)
                
                with st.chat_message("assistant"):
                    # Get the response and update the message history from agent
                    answer, message_history = asking_agent(model, query, system_message, st.session_state["message_history"])
                    st.markdown(answer)
                # Store the conversation history in session state
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.session_state["message_history"] = message_history

        except ResourceExhausted as e:
            st.error("Resource Exhausted: The request exceeded the available resources. Please try again later.", icon="🚨")
            st.error(f"Details: {str(e)}")
            logger.error(f"ResourceExhausted: {str(e)}")
        except Exception as e:
            st.error("An unexpected error occurred.", icon="🚨")
            st.error(f"Details: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        st.error("An unexpected error occurred in the main function.", icon="🚨")
        st.error(f"Details: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}")
        traceback.print_exc()

# ------------------------------------------------------------
# ★★★★★★  functions ★★★★★★
# ------------------------------------------------------------
def asking_agent(model, query, system_message, message_history):
    langgraph_agent_executor = create_react_agent(model, tools, state_modifier=system_message)
    # 1st time for asking agent
    if message_history == []:
        messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
    else:
        messages = langgraph_agent_executor.invoke(
            {"messages": message_history + [("human", query)]}
        )
    answer = messages["messages"][-1].content
    message_history = messages["messages"]

    return answer, message_history

# ------------------------------------------------------------
# ★★★★★★  execution part  ★★★★★★
# ------------------------------------------------------------
if __name__ == '__main__':

    # execute
    main()