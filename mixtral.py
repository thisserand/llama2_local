from transformers import AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
import gradio as gr
import transformers
import torch


def format_history(history, pipeline) -> str:
    # history has the following structure:
    # - dialogs
    # --- instruction
    # --- response (None for the most recent dialog)
    messages = []

    # add historic dialogs
    for dialog in history[:-1]:
      instruction, response = dialog[0], dialog[1]
      messages.extend([{"role": "user", "content": instruction},
                       {"role": "assistant", "content": response}])
    # add new instruction  
    new_instruction = history[-1][0].strip()
    messages.append({"role": "user", "content": new_instruction})
    
    # format messages list to prompt
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def load_generation_pipeline():
   model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
   
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)

   bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16
   )
   
   pipeline = transformers.pipeline(
      "text-generation",
      model=model_id,
      model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True, "quantization_config": bnb_config},
      streamer=streamer
   )
   return pipeline, streamer

def launch_ui(generation_pipeline, streamer):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            prompt = format_history(history, pipeline)

            history[-1][1] = ""
            kwargs = dict(text_inputs=prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            thread = Thread(target=pipeline, kwargs=kwargs)
            thread.start()

            for token in streamer:
                history[-1][1] += token
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.queue()
    demo.launch(share=True, debug=True)

if __name__ == '__main__':
  pipeline, streamer = load_generation_pipeline()
  launch_ui(pipeline, streamer)