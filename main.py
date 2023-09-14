import os
import time
import question_and_answer as qa

# Use this block if you want to upload a new document (via url or wikipedia) to Pinecone
# to be used with the LLM

# data = qa.load_document(os.getenv("QA_DOCUMENT"))
# data = qa.load_from_wikipedia(os.getenv("QA_WIKIPEDIA"))
# chunks = qa.chunk_data(data)
# qa.print_embedding_costs(chunks)
# qa.delete_pinecode_index()
# vector = qa.insert_or_fetch_embeddings(os.getenv("PINECONE_INDEX"), chunks)

# Use this line if the document that you want to the LLM to use is already uploaded into Pinecone
vector = qa.insert_or_fetch_embeddings(os.getenv("PINECONE_INDEX"))

question_counter = 1
print('Write Quit or Exit to quit')
while True:
    question = input(f'Question #{question_counter}: ')
    question_counter = question_counter + 1
    if question.lower() in ['quit', 'exit']:
        print('Quitting ... bye bye!')
        time.sleep(2)
        break
    result, chat_history = qa.ask_with_memory(vector, question)
    answer = result['answer']
    print(f'\nAnswer: {answer}')
    print(f'\n {"-" * 50} \n')
