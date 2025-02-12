from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import pinecone
import os

class ChatBot():
    load_dotenv() # Load environment variables from .env file
    
    # Load and split the documents
    loader = PyMuPDFLoader('./LosHechosDeLosApostolesEnAmerica.pdf') # The PDF file to be loaded.
    documents = loader.load() # Load data into Document objects.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4) # Split the documents into chunks
    docs = text_splitter.split_documents(documents) # Split the documents into chunks

    # Initialize HuggingFaceEmbeddings: 
    # HuggingFaceEmbeddings is a class that provides embeddings for text data using the Hugging Face Transformers library.
    embeddings = HuggingFaceEmbeddings()

    #Initialize Pinecone Client
    pc = pinecone.Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
    
    #Define Index name
    index_name = 'quickstart'

    #Checking index status
    if index_name not in pc.list_indexes().names():
        #Create new index
        pc.create_index(
            name = index_name, #The name of the index to create.
            metric = 'cosine', #The metric used to measure the similarity between vectors.
            dimension = 768, #The dimension of vectors that will be inserted in the index.
            spec= pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) #A dictionary containing configurations describing how the index should be deployed.
        )
        
        #Return VectorStore initialized from documents and embeddings
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        # Load pinecone vectorstore from index name.
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    #Define the repo ID and connect to gpt2 model on HuggingFace
    #Endpoint URL to use. If repo_id is not specified.
    llm = HuggingFaceEndpoint(
        endpoint_url = "https://api-inference.huggingface.co/models/gpt2",
        max_new_tokens=2048,#Maximum number of generated tokens
        top_k=10,#The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p=0.95,#If set to < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        typical_p=0.95,
        temperature=0.01,#The value used to module the logits distribution.
        repetition_penalty=1.03,#The parameter for repetition penalty. 1.0 means no penalty. 
        huggingfacehub_api_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),#The API token used to access the Hugging Face Hub.
        task="text-generation"#Task to call the model with. Should be a task that returns generated_text or summary_text.
    )

    #Template: Defines who the bot we are creating is
    template = """
    You are a historian dedicated to uncovering the truth, aiming to clarify the historical facts of Christianity in America and the era of colonization. Your goal is to help people understand these events through factual evidence.

    Users will ask you questions about Christianity, the Catholic Counter-Reformation, and the true history of colonization in America. Use the provided context to answer their questions. If you do not know the answer, simply state that you do not know.

    Respond concisely in two sentences.

    Context: {context}
    Question: {question}
    Answer: 

    
    """

    #Prompt template for a language model.
    #A prompt template consists of a string template.
    #It accepts a set of parameters from the user that can be used to generate a prompt for a language model.
    prompt = PromptTemplate(
        template = template,
        input_variables = ["context", "question"]
    )

    #The RAG chain is a chain that uses a retriever to retrieve relevant documents from a vector store and a language model to generate a response.
    #RunnablePassthrough: Create a new model by parsing and validating input data from keyword arguments.
    #StrOutputParser: A parser that parses the output of a language model into a string.
    #Prompt template for a language model and generat a response using the RAG chain.
    #llm: The language model to use for generating the response.
    rag_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )