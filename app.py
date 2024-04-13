import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import util, SentenceTransformer
import torch
import time
from time import perf_counter as timer
import textwrap
import json
import textwrap

import gradio as gr

print("Launching")

client = OpenAI()

# Define helper function to print wrapped text 
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Import saved file and view
embeddings_df_save_path = "./SRD_embeddings.csv"
print("Loading embeddings.csv")
text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
print("Embedding file loaded")
embedding_model_path = "BAAI/bge-m3"
print("Loading embedding model")
embedding_model = SentenceTransformer(model_name_or_path=embedding_model_path, 
                                      device='cpu') # choose the device to load the model to

# Convert the stringified embeddings back to numpy arrays
text_chunks_and_embedding_df_load['embedding'] = text_chunks_and_embedding_df_load['embedding_str'].apply(lambda x: np.array(json.loads(x)))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df_load.to_dict(orient="records")

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df_load["embedding"].tolist()), dtype=torch.float32).to('cpu')

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
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"File of Origin: {pages_and_chunks[index]['file_path']}")
        print("\n")

def prompt_formatter(query: str, 
                     context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Now use the following context items to answer the user query: {context}
    User query: {query}
    Answer:"""

    # Update base prompt with context items and query   
    

    
    return base_prompt.format(context=context, query=query)

system_prompt = """You are a game design expert specializing in Dungeons & Dragons 5e, answering beginner questions with descriptive, clear responses. Provide a story example. Avoid extraneous details and focus on direct answers. Use the examples provided as a guide for style and brevity. When responding:

    1. Identify the key point of the query.
    2. Provide a straightforward answer, omitting the thought process.
    3. Avoid additional advice or extended explanations.
    4. Answer in an informative manner, aiding the user's understanding without overwhelming them.
    5. DO NOT SUMMARIZE YOURSELF. DO NOT REPEAT YOURSELF.
    6. End with a line break and "What else can I help with?" 

Refer to these examples for your response style:

Example 1:
Query: How do I determine what my magic ring does in D&D?
Answer: To learn what your magic ring does, use the Identify spell, take a short rest to study it, or consult a knowledgeable character. Once known, follow the item's instructions to activate and use its powers.

Example 2:
Query: What's the effect of the spell fireball?
Answer: Fireball is a 3rd-level spell creating a 20-foot-radius sphere of fire, dealing 8d6 fire damage (half on a successful Dexterity save) to creatures within. It ignites flammable objects not worn or carried.

Example 3:
Query: How do spell slots work for a wizard?
Answer: Spell slots represent your capacity to cast spells. You use a slot of equal or higher level to cast a spell, and you regain all slots after a long rest. You don't lose prepared spells after casting; they can be reused as long as you have available slots.

Use the context provided to answer the user's query concisely. """



with gr.Blocks() as RulesLawyer:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):

        # Get relevant resources
        scores, indices = retrieve_relevant_resources(query=message,
                                                    embeddings=embeddings)
            
        # Create a list of context items
        context_items = [pages_and_chunks[i] for i in indices]

        # Format prompt with context items
        prompt = prompt_formatter(query=message,
                                context_items=context_items)
        print(prompt)
        bot_message = client.chat.completions.create(            
                        model="gpt-4",
                        messages=[
                            {
                            "role": "user",
                            "content": f"{system_prompt} {prompt}"
                            }
                        ],
                        temperature=1,
                        max_tokens=512,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
        chat_history.append((message, bot_message.choices[0].message.content))
        time.sleep(2)
        return "", chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    RulesLawyer.launch()