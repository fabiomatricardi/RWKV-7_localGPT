<img src='https://i.ytimg.com/vi/B3Qa2rRsaXo/maxresdefault.jpg' width=800>

# RWKV-7_localGPT
Local Gradio Chatbot with documents and RWKV-7 

Download the model *RWKV-v7-1.5B-World-v3* in GGUF format [from here](https://huggingface.co/zhiyuan8/RWKV-v7-1.5B-World-v3-GGUF)

Download the [latest binaries of llama.cpp](https://github.com/ggml-org/llama.cpp/releases) in the same directory

### Dependencies
```
pip install gradio pypdf tiktoken openai
```

## Run the interface and the model
from the terminal run
```
python gr_RWKV7_chat.py
```


### prompt guidelines
Read more in the [official blog post](https://wiki.rwkv.com/RWKV-Prompts/prompt-guidelines.html)


### The running app


<img src='https://github.com/fabiomatricardi/RWKV-7_localGPT/raw/main/RWKV7gradioDocs.gif' width=1000>

