# LLM Powered Q&A App

## Usage

This application can be used to upload external information, such as proprietary specific documents,and integrate them with Chat GPT's LLM (Large Language Model), so that the user can ask specific questions related to the uploaded documents but get a natural language response.

Currently, you need to manipulate the lines of code in the main.py file (via commenting out/uncommenting) so that you can switch between uploading documents or using an index that already has the uploaded documents.

## Process

The process of uploading your document to the program and it then being used for Q&As is the following:

1. Prepare the document (once per document)
    - Load the data into LangChain documents
    - Split the document into chunks
        - This helps optimize the relevance of the content we get back from a vector DB
        - Rule of thumb: If chunk text makes sense to a human without relevant context, it will make sense to a language model as well
    - Embed the chunks into numeric vectors
    - Save the chunks and embeddings to a vector db
2. Search (once per query)
    - Embed the user's question
    - Using the question's embedding and the chunk embedding, rank the vectors by similarity to the question's embedding
    - The nearest vector represents chunks similar to the question
3. Ask (once per query)
    - Insert the question and the most relevant chunks into a message to a GPT model
    - Return GPT's answer

### Environment Variables

Upon cloning the repository, you need to create a .env file and use the same name of the variables found in the .env.example file.
Here is a list of the variables and short description of what they represent:
- OPENAI_API_KEY - represents the API key associated with your OpenAI account
- PINECONE_API_KEY - represents the API key associated with your Pinecone account
- PINECONE_ENV - represents the environment associated with the Pinecone index that is holding the data of the document we want to used with the LLM 
- PINECONE_INDEX - represents the name of the Pinecone index that is holding the data of the document we want to used with the LLM 
- QA_DOCUMENT - the file path or url for the document we want to upload 
- QA_WIKIPEDIA - the search topic that we want to use to retrieve relevant documents from wikipedia so that they can be used with the LLM
