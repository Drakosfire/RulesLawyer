from openai import OpenAI

client = OpenAI()

def summarize_page(page_content):
    print(f"Summarizing page: {page_content}")
    if page_content == "":
        return ""
    page_system_prompt = "These are the text entries from a single page of a document. Please parse any messy text and concisely summarize the page. The summary should be focused on the critical contents of the page such as the plot, characters, setting, and important details or mechanics."
    
    page_summary_message = client.chat.completions.create(            
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"{page_system_prompt} {page_content}"
        }],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0  
    )
    return page_summary_message.choices[0].message.content

def summarize_document(all_summaries):
    document_summary_prompt = "Please concisely summarize the following text. The text is a compilation of summaries of individual pages from a document. The summaries are delimited by double newlines."      
    
    document_summary_message = client.chat.completions.create(            
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"{document_summary_prompt} {all_summaries}"
        }],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0  
    )
    return document_summary_message.choices[0].message.content