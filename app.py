import os
import gc
import torch
import streamlit as st
from PIL import Image

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(
    page_title="Document Image to Markdown Generator",
    layout="wide"
)

base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
local_adapter_path = "final_markdown_model"
online_adapter_path = os.environ.get("ADAPTER_MODEL_NAME", "")

pic_size = 512
new_words = 512

@st.cache_resource
def load_everything():
    if os.path.exists(local_adapter_path):
        adapter_path = local_adapter_path
    elif online_adapter_path.strip() != "":
        adapter_path = online_adapter_path
    else:
        adapter_path = local_adapter_path

    try:
        processor = AutoProcessor.from_pretrained(
            adapter_path,
            trust_remote_code=True
        )
    except:
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

    if torch.cuda.is_available():
        four_bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=four_bit,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    model.eval()

    return processor, model, adapter_path

def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def make_small_image(img):
    w, h = img.size

    if max(w, h) <= pic_size:
        return img

    if w > h:
        new_w = pic_size
        new_h = int(h * pic_size / w)
    else:
        new_h = pic_size
        new_w = int(w * pic_size / h)

    return img.resize((new_w, new_h))

def make_chat():
    line = "Convert this document image into clean Markdown. Keep headings, tables, lists, equations and reading order."

    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": line}
            ]
        }
    ]

    return chat

def move_batch(batch, model):
    model_device = next(model.parameters()).device
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(model_device)
        else:
            moved[key] = value

    return moved

def create_markdown(img):
    processor, model, adapter_path = load_everything()

    img = img.convert("RGB")
    img = make_small_image(img)

    chat = make_chat()

    text = processor.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )

    batch = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt"
    )

    batch = move_batch(batch, model)

    with torch.no_grad():
        ids = model.generate(
            **batch,
            max_new_tokens=new_words,
            do_sample=False
        )

    new_ids = ids[:, batch["input_ids"].shape[1]:]

    ans = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    clear_ram()

    return ans.strip()

st.title("Document Image to Markdown Generator")
st.write("Fine-tuned Qwen2-VL with QLoRA adapter for document image to Markdown generation.")

try:
    processor, model, adapter_path = load_everything()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Model loading failed.")
    st.exception(e)

with st.sidebar:
    st.header("Model Settings")
    st.write("Base model")
    st.code(base_model_name)
    st.write("Adapter")
    if online_adapter_path.strip() != "":
        st.code(online_adapter_path)
    else:
        st.code(local_adapter_path)
    st.write("Image size")
    st.code(str(pic_size))
    st.write("Max new tokens")
    st.code(str(new_words))
    if torch.cuda.is_available():
        st.success("GPU available")
    else:
        st.warning("CPU mode. This can be very slow.")

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Generated Markdown")

        if st.button("Generate Markdown", disabled=not model_ready):
            with st.spinner("Generating markdown..."):
                try:
                    result = create_markdown(image)

                    st.text_area(
                        "Markdown Output",
                        value=result,
                        height=500
                    )

                    st.download_button(
                        label="Download Markdown",
                        data=result,
                        file_name="generated_markdown.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error("Generation failed.")
                    st.exception(e)
else:
    st.info("Upload an image to generate Markdown.")
