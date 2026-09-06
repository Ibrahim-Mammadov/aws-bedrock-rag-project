import boto3
import json
import os
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "eu-central-1")
)

# 2. Load the Bedrock Embeddings model
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
)

# 3. Load the local FAISS vector database
vector_store = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 4. Define the question you want to ask
question = "What foundation models does Amazon Bedrock support?"

# 5. Search the vector database for relevant information (Retrieval)
print("Searching the database for answers...\n")
docs = vector_store.similarity_search(question, k=2)
context_text = "\n".join([doc.page_content for doc in docs])

# 6. Build the prompt for the AI (Augmented)
prompt = f"""You are a helpful and professional assistant. Use the following context to answer the user's question. 

Context:
{context_text}

Question:
{question}
"""

# 7. Prepare the payload for Amazon Nova model
payload = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ],
    "inferenceConfig": {
        "max_new_tokens": 300,
        "temperature": 0.1
    }
}

# 8. Send the request to Amazon Bedrock (Generation)
try:
    response = bedrock_client.invoke_model(
        modelId="eu.amazon.nova-lite-v1:0",
        body=json.dumps(payload)
    )
    
    result = json.loads(response['body'].read())
    answer = result['output']['message']['content'][0]['text']
    
    print("--- Question ---")
    print(question)
    print("\n--- AI Answer ---")
    print(answer)

except Exception as e:
    print(f"Error occurred: {e}")
