import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import util, SentenceTransformer
import torch
import time
from time import perf_counter as timer
from datetime import datetime
import textwrap
import json
import gradio as gr

print("Launching")

client = OpenAI()

# Load the enhanced JSON file with summaries
def load_enhanced_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

enhanced_json_file = "Swords&Wizardry_enhanced_output.json"
enhanced_data = load_enhanced_json(enhanced_json_file)

# Extract document summary and page summaries
document_summary = enhanced_data.get('document_summary', 'No document summary available.')
page_summaries = {int(page): data['summary'] for page, data in enhanced_data.get('pages', {}).items()}

# Import saved file and view
embeddings_df_save_path = "Swords&Wizardry_output_embeddings.csv"
print("Loading embeddings.csv")
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
print("Embedding file loaded")

# Convert the stringified embeddings back to numpy arrays
text_chunks_and_embedding_df_load['embedding'] = text_chunks_and_embedding_df_load['embedding_str'].apply(lambda x: np.array(json.loads(x)))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df_load.to_dict(orient="records")

# Debug: Print the first few rows and column names
print("DataFrame columns:", text_chunks_and_embedding_df_load.columns)
print("\nFirst few rows of the DataFrame:")
print(text_chunks_and_embedding_df_load.head())

# Debug: Print the first item in pages_and_chunks
# print("\nFirst item in pages_and_chunks:")
# print(pages_and_chunks[0])

embedding_model_path = "BAAI/bge-m3"
print("Loading embedding model")
embedding_model = SentenceTransformer(model_name_or_path=embedding_model_path, 
                                      device='cpu') # choose the device to load the model to

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df_load["embedding"].tolist()), dtype=torch.float32).to('cpu')

# Define helper function to print wrapped text 
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def hybrid_estimate_tokens(text: str)-> float:
    # Part 1: Estimate based on spaces and punctuation
    estimated_words = text.count(' ') + 1  # Counting words by spaces
    punctuation_count = sum(1 for char in text if char in ',.!?;:')  # Counting punctuation as potential separate tokens
    estimate1 = estimated_words + punctuation_count
    
    # Part 2: Estimate based on total characters divided by average token length
    average_token_length = 4
    total_characters = len(text)
    estimate2 = (total_characters // average_token_length) + punctuation_count
    
    # Average the two estimates
    estimated_tokens = (estimate1 + estimate2) / 2

    return estimated_tokens


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=4,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    print(f"Number of results: {len(indices)}")
    print(f"Indices: {indices}")
    print(f"Total number of chunks: {len(pages_and_chunks)}")
    
    for i, (score, index) in enumerate(zip(scores, indices)):
        print(f"\nResult {i+1}:")
        print(f"Score: {score:.4f}")
        print(f"Index: {index}")
        
        if index < 0 or index >= len(pages_and_chunks):
            print(f"Error: Index {index} is out of range!")
            continue
        
        chunk = pages_and_chunks[index]
        print(f"Token Count: {chunk['chunk_token_count']}")
        print("Available keys:", list(chunk.keys()))
        print("sentence_chunk content:", repr(chunk.get("sentence_chunk", "NOT FOUND")))
        
        chunk_text = chunk.get("sentence_chunk", "Chunk not found")
        print_wrapped(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
        
        print(f"File of Origin: {chunk['file_path']}")

    return scores, indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    # Include document summary
    formatted_context = f"Document Summary: {document_summary}\n\n"

    # Add context items with their page summaries
    for item in context_items:
        page_number = item.get('page', 'Unknown')
        page_summary = page_summaries.get(page_number, 'No page summary available.')
        formatted_context += f"Summary: {page_summary}\n"
        formatted_context += f"Content: {item['sentence_chunk']}\n\n"

    base_prompt = """Use the following context to answer the user query:

{context}

User query: {query}
Answer:"""
    print(f"Prompt: {base_prompt.format(context=formatted_context, query=query)}")
    return base_prompt.format(context=formatted_context, query=query)

system_prompt = """You are a friendly and technical answering system, answering questions with accurate, grounded, descriptive, clear, and specific responses. ALWAYS provide a page number citation. Provide a story example. Avoid extraneous details and focus on direct answers. Use the examples provided as a guide for style and brevity. When responding:

    1. Identify the key point of the query.
    2. Provide a straightforward answer, omitting the thought process.
    3. Avoid additional advice or extended explanations.
    4. Answer in an informative manner, aiding the user's understanding without overwhelming them or quoting the source.
    5. DO NOT SUMMARIZE YOURSELF. DO NOT REPEAT YOURSELF. 
    6. End with page citations, a line break and "What else can I help with?" 

    Example:
    Query: Explain how the player should think about balance and lethality in this game. Explain how the game master should think about balance and lethality?
    Answer: In "Swords & Wizardry: WhiteBox," players and the game master should consider balance and lethality from different perspectives. For players, understanding that this game encourages creativity and flexibility is key. The rules are intentionally streamlined, allowing for a potentially high-risk environment where player decisions significantly impact outcomes. The players should think carefully about their actions and strategy, knowing that the game can be lethal, especially without reliance on intricate rules for safety. Page 33 discusses the possibility of characters dying when their hit points reach zero, although alternative, less harsh rules regarding unconsciousness and recovery are mentioned.

For the game master (referred to as the Referee), balancing the game involves providing fair yet challenging scenarios. The role of the Referee isn't to defeat players but to present interesting and dangerous challenges that enhance the story collaboratively. Page 39 outlines how the Referee and players work together to craft a narrative, with the emphasis on creating engaging and potentially perilous experiences without making it a zero-sum competition. Referees can choose how lethal the game will be, considering their group's preferred play style, including implementing house rules to soften deaths or adjust game balance accordingly.

Pages: 33, 39

Use the context provided to answer the user's query concisely. """

with gr.Blocks() as RulesLawyer:

    message_state = gr.State()
    chatbot_state = gr.State([])
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def store_message(message):           
        return message       

    def respond(message, chat_history):
        print(datetime.now())
        print(f"User Input : {message}")
        print(f"Chat History: {chat_history}")
        print(f"""Token Estimate: {hybrid_estimate_tokens(f"{message} {chat_history}")}""")

        # Get relevant resources
        scores, indices = print_top_results_and_scores(query=message,
                                                    embeddings=embeddings)
                    
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]

        # Format prompt with context items
        prompt = prompt_formatter(query=f"Chat History : {chat_history} + {message}",
                                  context_items=context_items)
        
        bot_message = client.chat.completions.create(            
                        model="gpt-4o",
                        messages=[
                            {
                            "role": "user",
                            "content": f"{system_prompt} {prompt}"
                            }
                        ],
                        temperature=1,
                        max_tokens=1000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
        chat_history.append((message, bot_message.choices[0].message.content))
        print(f"Response : {bot_message.choices[0].message.content}")
        
        time.sleep(2)
        return "", chat_history
    msg.change(store_message, inputs = [msg], outputs = [message_state])
    chatbot.change(store_message, [chatbot], [chatbot_state])
    msg.submit(respond, [message_state, chatbot_state], [msg, chatbot])

if __name__ == "__main__":
    RulesLawyer.launch()
