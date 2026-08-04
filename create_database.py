import boto3
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-central-1"
)

# 2. Load the text document
loader = TextLoader("document.txt")
documents = loader.load()

# 3. Split the text into smaller chunks for the AI to process easily
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 4. Create Bedrock Embeddings (Convert text to vectors)
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
)

# 5. Create a FAISS vector database and save it locally
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")

print("Vector database created successfully!")