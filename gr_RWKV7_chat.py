# from https://huggingface.co/spaces/ysharma/Microsoft_Phi-3-Vision-128k/blob/main/app.py
# https://wiki.rwkv.com/


import gradio as gr
import base64
from openai import OpenAI
from PIL import Image
import io
from datetime import datetime
import random
import string
import subprocess
import os
import signal
from time import sleep
import threading
import pypdf
import tiktoken


def countTokens(text):
    if text is None: return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    numoftokens = len(encoding.encode(str(text)))
    return numoftokens

def PDFtoText(pdffile):
    # try to read the PDF and write it into a txt file, same name but .txt extension
    try:
        reader = pypdf.PdfReader(pdffile)
        text = ""
        page_count = len(reader.pages)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        a = text.strip()
        textfile = a.replace('\n\n','')
        print('Creating text file...')
        gr.Info(f"Parsed from PDF {page_count} pages of text\nA total context of {countTokens(textfile)} tokens ")
        return textfile
    except Exception as e:
        gr.Error(f"Error reading PDF {pdffile}: {e}")

# when using llamacpp-server, you need to check if the stream chunk is present
# usually the first and the last chunk are empty and will throw an error
# https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks

# Global variable to store the process

RWKVserver = None

# Background of the Chatbot as a placeholder... really smart!
PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://i.ytimg.com/vi/B3Qa2rRsaXo/maxresdefault.jpg" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">RWKV 7 Goose</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">The best of RNN and Transformer, constant memory usage, constant inference generation speed, infinite context length.</p>
</div>
"""

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    print(f'Logfile_{res}.md  CREATED')
    return f'Logfile_{res}.md'



# CSS only to justify center the title
mycss = """
#warning {justify-content: center; text-align: center}
"""
# warmup prompt
prompt = """User: Hello.

Assistant:"""

logafilename = genRANstring(5)

with gr.Blocks(theme=gr.themes.Soft(secondary_hue=gr.themes.colors.orange,primary_hue=gr.themes.colors.blue),
               fill_width=True,css=mycss) as demo: #https://www.gradio.app/guides/theming-guide
        gr.Markdown("# Chat with RWKV 7 Goose - 1.5b RNN in context learner",elem_id='warning')
        with gr.Row():
            with gr.Column(scale=1):
                genlogo = gr.Image('https://i.ibb.co/fYG0LR21/rwkv7logo.png',
                                   show_label=False)
                gr.Markdown('### Server Section')
                start_btn = gr.Button("Start Model Server",variant='primary')
                stop_btn = gr.Button("Stop Model Server") 
                srv_stat = gr.Textbox(label="Status")                      
                APIKey = gr.Textbox(value="not required", 
                            label="Open Router API key",
                            type='password',visible=False)
                with gr.Accordion(open=False, label='Tuning Parameters'):
                    maxlen = gr.Slider(minimum=250, maximum=4096, value=2048, step=1, label="Max new tokens")
                    temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.45, step=0.1, label="Temperature")          
                    log = gr.Markdown(logafilename, label='Log File name',container=True, show_label=True)
                    botgr = gr.TextArea(value='',show_label=True,visible=False) #segregate chat message from presentation
                    firstTurn = gr.Checkbox(value=True, visible=False)
                closeall = gr.Button("Close the app",variant='primary')
                gr.Markdown('---')
                clear = gr.Button(value='Delete History',variant='secondary',
                                  icon='https://img.freepik.com/premium-vector/minimal-trash-bin-icon-vector_941526-16016.jpg')
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages",min_height='72vh',
                                     placeholder=PLACEHOLDER,
                                     show_copy_button = True,
                                     avatar_images=['https://i.ibb.co/PvqbDphL/user.png',
                                     'https://hackernoon.imgix.net/images/PAvDPagOxFNLfOhJ2d7DslfLJV03-t8a30q8.png'],)
                msg = gr.MultimodalTextbox(interactive=True, file_types=[".pdf"], 
                                           placeholder="Enter message or upload file...", 
                                           show_label=False)

                def clearData():
                    hiddenChat = gr.TextArea(value="",show_label=True,visible=False)
                    return hiddenChat

                def user(user_message, history, cbthst,firstturn):
                    if  user_message['files']:
                        print('we have a pdf file')
                        pdffile = user_message['files'][-1]
                        text = user_message['text']
                        CTX_text = PDFtoText(pdffile)
                        logging = f'USER PDF> {CTX_text}\nUSER text> {text}\n'
                        writehistory(logafilename,logging)
                        print(logging)
                        #goes to the chatbox for presentation only
                        cbthst.append({"role": "user", "content": gr.File(value=pdffile,)})
                        cbthst.append({"role": "user", "content": text})
                        #goes to the API // but add 2 lines if this is not the first turn
                        if firstturn == True:
                            history = history + 'User: Read the provided text and reply to the question\nQhestion: '+text+'\nProvided text: '+ CTX_text+'\n\nAssistant:'    
                            firstturn = False
                        else:
                            history = history + '\n\nUser: Read the provided text and reply to the question\nQhestion: '+text+'\nProvided text: '+ CTX_text+'\n\nAssistant:'
                    else:
                        print('no pdf included')
                        text = user_message['text']
                        logging = f'USER text> {text}\n'
                        writehistory(logafilename,logging)
                        print(logging)
                        cbthst.append({"role": "user", "content": text})   #goes to the chatbox for presentation only
                        if firstturn == True:
                            history = history + 'User: '+ text+'\n\nAssistant:'#goes to the API   
                        else:    
                            history = history + '\n\nUser: '+ text+'\n\nAssistant:'#goes to the API                           
                    return "", history, cbthst, firstturn
                        

                def respond(chat_history, api,t,m,cbthst):
                    print(cbthst)
                    plaintext =''
                    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
                    stream = client.completions.create(
                        prompt=cbthst,
                        model="RWKV-7",    
                        max_tokens=m,
                        stream=True,
                        temperature=t,
                        stop=['Q:','User:'])
                    chat_history.append({"role": "assistant", "content": ""})
                    for chunk in stream:
                        chat_history[-1]['content'] += chunk.choices[0].text
                        cbthst += chunk.choices[0].text
                        yield chat_history, cbthst
                    logging = f"ASSISTANT> {chat_history[-1]['content']}\n"
                    writehistory(logafilename,logging)  
                    if not cbthst.endswith('\n\n'):
                        cbthst += '\n\n'   
                    yield chat_history, cbthst              

                ################# FUNCTIONS TO START/STOP SERVER AND WARMUP PROMPT ######################
                def start_server():
                    global RWKVserver
                    
                    # Check if process is already running
                    if RWKVserver is not None and RWKVserver.poll() is None:
                        return "Server is already running!"
                    
                    # Start the server process - add Repeat penalty and --dry-multiplier for long context
                    RWKVserver = subprocess.Popen([
                        'llama-server.exe',
                        '-m',
                        'rwkv7-1.5B-world-Q5_K_M.gguf',
                        '-c',         
                        '51200',
                        '-ngl',
                        '0',
                        '--repeat-penalty',
                        '1.7',
                        '--dry-multiplier',
                        '0.8',   
                        '--port',
                        '8080',       
                    ], creationflags=subprocess.CREATE_NEW_CONSOLE)
                    
                    return f"Server started with PID: {RWKVserver.pid}"

                def delayed_exit():
                    # Give the interface time to send the response before exiting
                    import time
                    time.sleep(2)
                    os._exit(0)  # Force exit the Python process

                def stop_server():
                    global RWKVserver
                    
                    if RWKVserver is None:
                        return "No server is running!"
                    
                    if RWKVserver.poll() is None:  # Check if process is still running
                        try:
                            RWKVserver.terminate()
                            # Wait for process to terminate (optional)
                            RWKVserver.wait(timeout=5)
                            return "Server stopped successfully!"
                        except subprocess.TimeoutExpired:
                            RWKVserver.kill()
                            return "Server killed forcefully!"
                        except Exception as e:
                            return f"Error stopping server: {str(e)}"
                    else:
                        return "Server is not running!"

                def init_shutdown():
                    # Schedule a forced exit after showing the message
                    threading.Thread(target=delayed_exit, daemon=True).start()
                    return " Closing Gradio interface..."
                ##################### SERVER SIDE COMPLETED #############################################

        # HANDLE SERVER EVENTS
        start_btn.click(start_server, inputs=[], outputs=srv_stat)
        stop_btn.click(stop_server, inputs=[], outputs=srv_stat) 
        closeall.click(stop_server, inputs=[], outputs=srv_stat).then(init_shutdown,inputs=[], outputs=srv_stat)   
        # HANDLE CHAT INTERFACE
        clear.click(clearData,[],[botgr])
        msg.submit(user, [msg, botgr,chatbot], [msg, botgr, chatbot]).then(
                respond, [chatbot,APIKey,temperature,maxlen,botgr], [chatbot,botgr])


# RUN THE MAIN
if __name__ == "__main__":
    demo.launch(inbrowser=True)

