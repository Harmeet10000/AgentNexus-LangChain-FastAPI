from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# Create a traceable function
@traceable(run_type="chain", name="Gemini QA")
def ask_gemini(question: str):
    """Ask Gemini a question and get a response."""
    messages = [{"role": "user", "content": question}]
    response = model.invoke(messages)
    return response.content


# Use the function - this automatically traces to LangSmith
result = ask_gemini("What is LangChain?")
print(result)
