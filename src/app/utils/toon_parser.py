# Imagine a community library 'toon_python' exists in 2026
from typing import Any

import toons
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate


class ToonParser(BaseOutputParser):
    def parse(self, text: str) -> Any:

        return toons.dumps(text)


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

# Write to file
# with open("data.toon", "w") as f:
#     toons.dump({"message": "Hello, TOON!"}, f)

# # Read from file
# with open("data.toon", "r") as f:
#     data = toons.load(f)
