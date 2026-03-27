# Imagine a community library 'toon_python' exists in 2026
import toons
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate


class ToonParser(BaseOutputParser):
    def parse(self, text: str):
        return decode_toon(text)


# 1. Instruct the model to use TOON
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a data extractor. Output ONLY in TOON format."),
        ("human", "Extract these users: {input}"),
    ]
)

# 2. Chain it together
# chain = prompt | llm | ToonParser()

# 3. Output will be a clean Python Dict
# result = chain.invoke({"input": "Alice (ID 1), Bob (ID 2)"})
