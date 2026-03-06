import toons
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Prepare your data
data = [
    {"id": 1, "name": "Harmeet", "role": "Backend"},
    {"id": 2, "name": "Gemini", "role": "AI"}
]

# 2. Convert to TOON format (saves ~40% tokens vs JSON)
toon_data = toons.dumps(data)

# 3. Use in LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
prompt = ChatPromptTemplate.from_template(
    "Analyze the following user data provided in TOON format and summarize it:\n\n{data}"
)

chain = prompt | llm
response = chain.invoke({"data": toon_data})
print(response.content)
# If you are building a RAG (Retrieval-Augmented Generation) system, using TOON allows you to fit twice as many search results into the same context window compared to JSON, giving the AI more "knowledge" to work with before it starts forgetting.
# , make sure your System Prompt includes a brief instruction on what the TOON format looks like (e.g., Respond using Token Oriented Object Notation), otherwise the model might default back to JSON.
# you write a custom LangChain OutputParser that automatically handles TOON responses