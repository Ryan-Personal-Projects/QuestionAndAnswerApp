import os
from dotenv import load_dotenv
from langchain.document_loaders import WikipediaLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

def load_document(file):    ''
    """
    Takes in a string that represents either a file path or a url for a document and
    converts it into a LangChain document. Currently only supports PDF document types

    Parameters
    ----------
    file:
        A string that represents a file path or url to a document

    :returns
        The original document as a LangChain document
    """
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        # This will load the pdf into an array of documents, where each document contains the page content
        # and the metadata with a page number
        print(f'Loading {file}')
        # This can also load online pdfs, just enter url
        loader = PyPDFLoader(file)
    else:
        print('Document format is not supported')
        return None

    # Will return a list of langchain documents, one document per page
    data = loader.load()
    return data

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    """
    Takes in a search topic that is used to find documents from Wikipedia. Language and max number of documents
    are also optional parameters to designate the language of the document and the max number of documents
    that can be returned. These documents are than converted into a LangChain document.
    Parameters
    ----------
    query:
        A string that represents a search topic that is used to find documents from Wikipedia
    lang: 
        A string that represents an abbreviation of the desired language for the documents to be in
    load_max_docs:
        A number that represents the max number of documents that can be returned

    :returns
        The Wikipedia document as a LangChain document
    """
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    print(f'Loading "{query}" from wikipedia')
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    """
    Takes in data and splits it into chunks

    Parameters
    ----------
    data:
        An amount of data (ex: file)
    chunk_size:
        An integer representing the size of the chunks

    :returns
        The original data split into an array of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks
    
def insert_or_fetch_embeddings(index_name, chunks=''):
    """
    Takes in an index name that represents the name of an index in the designated Pinecone account,
    or the name of a new index that will be created. If a new index is being created, than the chunks data
    is a required parameter. The function will either return the vector store of an already existing index 
    or embed the chunks data into numeric vectors and store them in a newly created index and return the 
    vector store of it.

    Parameters
    ----------
    index_name:
        The name of the index associated with the Pinecone account, or the name of the new index to be created
        in Pinecone
    chunk:
        The data that will be embedded into numeric vectors

    :returns
        A vector store associated with the designated index
    """
    embeddings = OpenAIEmbeddings()

    #initiate Pinecone object
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
       print(f'Creating index {index_name} and embeddings...', end='') 
       pinecone.create_index(index_name, dimension=1536, metric='cosine')
       vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    
    print('Done')
    return vector_store

def delete_pinecode_index(index_name='all'):
    """
    Deletes indexes from the designated Pinecone account. If a specific index name is not
    given, than all indexes will be deleted from the account.

    Parameters
    ----------
    index_name:
        The name of the index to be deleted
    """
    #initiate Pinecone object
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes...')
        for index in indexes:
            pinecone.delete_index(index)
    else:
        print(f'Deleting index {index_name}...', end='')
        pinecone.delete_index(index_name)
    print('Done')

def ask_and_get_answer(vector_store, question):
    """
    Takes in a vector_store associated with a set of data and combines with Chat GPT's LLM, so that
    specific questions about that data can be asked and the LLM will be able to 
    answer in natual language responses.

    Parameters
    ----------
    vector_store:
        The vector store that holds the numeric vectors of the data that the user wants to teach the 
        LLM about
    question:
        A question that the user wants to ask the LLM

    :returns
        The LLM's answer to the user's question
    """
    #First retrieve most relevant chunks from vector db
    #And then feed relevant chunks to LLM for final answer 
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(question)
    return answer

def ask_with_memory(vector_store, question, chat_history=[]):
    """
    Takes in a vector_store associated with a set of data and combines with Chat GPT's LLM, so that
    specific questions about that data can be asked and the LLM will be able to 
    answer in natual language responses. It also takes into account chat history, so questions can be
    asked using information from previous responses.

    Parameters
    ----------
    vector_store:
        The vector store that holds the numeric vectors of the data that the user wants to teach the 
        LLM about
    question:
        A question that the user wants to ask the LLM
    chat_history:
        An array holding that data of previous question/answer responses

    :returns
        The LLM's answer to the user's question
    """
    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history
