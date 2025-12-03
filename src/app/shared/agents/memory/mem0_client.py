from mem0 import MemoryClient

client = MemoryClient(api_key="your-api-key")
# ========================================================
messages = [
  { "role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts." },
  { "role": "assistant", "content": "Hello Alex! I see that you're a vegetarian with a nut allergy." }
]

client.add(messages, user_id="alex")
# ========================================================
query = "What can I cook for dinner tonight?"

filters = {
 "OR":[
    {
       "user_id":"alex"
    }
 ]
}

client.search(query, version="v2", filters=filters)

# =====================================================
def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Retrieve relevant context from Mem0"""
    try:
        memories = mem0.search(query, user_id=user_id)
        memory_list = memories["results"]

        serialized_memories = " ".join([mem["memory"] for mem in memory_list])
        context = [
            {
                "role": "system",
                "content": f"Relevant information: {serialized_memories}",
            },
            {"role": "user", "content": query},
        ]
        return context
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        # Return empty context if there's an error
        return [{"role": "user", "content": query}]


def generate_response(input: str, context: List[Dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | llm
    response = chain.invoke({"context": context, "input": input})
    return response.content


def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to Mem0"""
    try:
        interaction = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]
        result = mem0.add(interaction, user_id=user_id)
        print(
            f"Memory saved successfully: {len(result.get('results', []))} memories added"
        )
    except Exception as e:
        print(f"Error saving interaction: {e}")

        
