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

