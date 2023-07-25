import os
import gradio as gr
import fire
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import TextIteratorStreamer
from llama_chat_format import format_to_llama_chat_style


# class syntax
class Model_Type(Enum):
    gptq = 1
    ggml = 2
    full_precision = 3


def get_model_type(model_name):
  if "gptq" in model_name.lower():
    return Model_Type.gptq
  elif "ggml" in model_name.lower():
    return Model_Type.ggml
  else:
    return Model_Type.full_precision


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
      model = AutoGPTQForCausalLM.from_quantized(model_name, device_map="auto", use_safetensors=True, use_triton=False)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=True)
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    return model, tokenizer


def init_auto_model_and_tokenizer(model_name, model_type, file_name=None):
  model_type = get_model_type(model_name)

  if Model_Type.ggml == model_type:
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    file_path = hf_hub_download(repo_id=model_name, filename=file_name, local_dir=models_folder)
    model = Llama(file_path, n_ctx=4096)
    tokenizer = None
  else:
    model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type=model_type)
  return model, tokenizer


def run_ui(model, tokenizer, is_chat_model, model_type):
  with gr.Blocks() as demo:
      chatbot = gr.Chatbot()
      msg = gr.Textbox()
      clear = gr.Button("Clear")

      def user(user_message, history):
          return "", history + [[user_message, None]]

      def bot(history):
          if is_chat_model:
              instruction = format_to_llama_chat_style(history)
          else:
              instruction =  history[-1][0]

          history[-1][1] = ""
          kwargs = dict(temperature=0.6, top_p=0.9)
          if model_type == Model_Type.ggml:
              kwargs["max_tokens"] = 512
              for chunk in model(prompt=instruction, stream=True, **kwargs):
                  token = chunk["choices"][0]["text"]
                  history[-1][1] += token
                  yield history

          else:
              streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
              inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
              kwargs["max_new_tokens"] = 512
              kwargs["input_ids"] = inputs["input_ids"]
              kwargs["streamer"] = streamer
              thread = Thread(target=model.generate, kwargs=kwargs)
              thread.start()

              for token in streamer:
                  history[-1][1] += token
                  yield history

      msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
      clear.click(lambda: None, None, chatbot, queue=False)
  demo.queue()
  demo.launch(share=True, debug=True)

def main(model_name=None, file_name=None):
    assert model_name is not None, "model_name argument is missing."

    is_chat_model = 'chat' in model_name.lower()
    model_type = get_model_type(model_name)

    if model_type == Model_Type.ggml:
      assert file_name is not None, "When model_name is provided for a GGML quantized model, file_name argument must also be provided."

    model, tokenizer = init_auto_model_and_tokenizer(model_name, model_type, file_name)
    run_ui(model, tokenizer, is_chat_model, model_type)

if __name__ == '__main__':
  fire.Fire(main)