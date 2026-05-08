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

# ── constants ────────────────────────────────────────────────────────────────
base_model_name      = "Qwen/Qwen2-VL-2B-Instruct"
local_adapter_folder = "final_markdown_model"          # only used when running locally
online_adapter_path  = os.environ.get("ADAPTER_MODEL_NAME", "").strip()  # e.g. "supremeproducts45/qwen_lora"

pic_size  = 512
new_words = 512


# ── helpers ──────────────────────────────────────────────────────────────────
def clear_ram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_small_image(img):
    w, h = img.size
    if max(w, h) <= pic_size:
        return img
    if w > h:
        new_w, new_h = pic_size, int(h * pic_size / w)
    else:
        new_h, new_w = pic_size, int(w * pic_size / h)
    return img.resize((new_w, new_h))


def has_adapter_files(folder: str) -> bool:
    """Return True if folder contains the required adapter files."""
    config   = os.path.join(folder, "adapter_config.json")
    safe     = os.path.join(folder, "adapter_model.safetensors")
    bin_file = os.path.join(folder, "adapter_model.bin")
    return os.path.isfile(config) and (os.path.isfile(safe) or os.path.isfile(bin_file))


def find_adapter_path():
    """
    Priority (Streamlit Cloud deployment):
      1. ADAPTER_MODEL_NAME env var  →  HuggingFace Hub repo ID  (primary on cloud)
      2. ./final_markdown_model/     →  local folder (useful when running locally)
      3. ./                          →  root folder  (adapter files copied next to app.py)
    """
    # 1. HuggingFace Hub repo set via Streamlit secrets / env var
    if online_adapter_path:
        return online_adapter_path

    # 2. Local sub-folder
    if has_adapter_files(local_adapter_folder):
        return local_adapter_folder

    # 3. Files placed right next to app.py
    if has_adapter_files("."):
        return "."

    return None


# ── model loading (cached so it runs only once per session) ──────────────────
@st.cache_resource
def load_everything(adapter_path: str):
    # --- processor -----------------------------------------------------------
    # Try loading from the adapter repo first (it may contain tokenizer files).
    # Fall back to the base model if the adapter repo has no processor config.
    try:
        processor = AutoProcessor.from_pretrained(
            adapter_path,
            trust_remote_code=True
        )
    except Exception as proc_err:
        st.warning(
            f"Processor not found in adapter repo ({proc_err}). "
            "Loading from base model instead."
        )
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

    # --- base model ----------------------------------------------------------
    if torch.cuda.is_available():
        four_bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=four_bit,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # --- LoRA adapter --------------------------------------------------------
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return processor, model


# ── inference ─────────────────────────────────────────────────────────────────
def make_chat():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Convert this document image into clean Markdown. "
                        "Keep headings, tables, lists, equations and reading order."
                    ),
                },
            ],
        }
    ]


def move_batch(batch, model):
    device = next(model.parameters()).device
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def create_markdown(img: Image.Image) -> str:
    adapter_path = find_adapter_path()

    if adapter_path is None:
        raise ValueError(
            "Adapter model not found.\n"
            "• Set ADAPTER_MODEL_NAME to your HuggingFace repo (e.g. supremeproducts45/qwen_lora) "
            "in Streamlit Cloud secrets, OR\n"
            "• Put adapter_config.json + adapter_model.safetensors inside the "
            f"'{local_adapter_folder}/' folder (local use only)."
        )

    processor, model = load_everything(adapter_path)

    img = img.convert("RGB")
    img = make_small_image(img)

    chat = make_chat()
    text = processor.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    batch = processor(
        text=[text],
        images=[img],
        padding=True,
        return_tensors="pt",
    )
    batch = move_batch(batch, model)

    with torch.no_grad():
        ids = model.generate(
            **batch,
            max_new_tokens=new_words,
            do_sample=False,
        )

    new_ids = ids[:, batch["input_ids"].shape[1]:]
    ans = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    clear_ram()
    return ans.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Document Image to Markdown Generator")
st.write("Fine-tuned Qwen2-VL with QLoRA adapter for document image to Markdown generation.")

adapter_path = find_adapter_path()

with st.sidebar:
    st.header("Model Settings")

    st.write("Base model")
    st.code(base_model_name)

    st.write("Adapter")
    if adapter_path is None:
        st.error("Adapter missing — set ADAPTER_MODEL_NAME in Streamlit secrets.")
    else:
        st.success("Adapter found")
        st.code(adapter_path)

    st.write("Image size")
    st.code(str(pic_size))

    st.write("Max new tokens")
    st.code(str(new_words))

    if torch.cuda.is_available():
        st.success("GPU available")
    else:
        st.warning("CPU mode — Qwen2-VL will be very slow without a GPU.")

if adapter_path is None:
    st.error("Adapter model not found. Configure it using one of the methods below.")
    st.markdown("""
**Method 1 — Streamlit Cloud (recommended)**  
Add a secret in your app's Settings → Secrets:
```toml
ADAPTER_MODEL_NAME = "supremeproducts45/qwen_lora"
```

**Method 2 — Local use**  
Place `adapter_config.json` + `adapter_model.safetensors` inside the
`final_markdown_model/` folder next to `app.py`.

**Method 3 — Local use (root)**  
Place the same files directly next to `app.py`.
""")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Generated Markdown")

        if st.button("Generate Markdown"):
            with st.spinner("Generating markdown…"):
                try:
                    result = create_markdown(image)
                    st.text_area("Markdown Output", value=result, height=500)
                    st.download_button(
                        label="Download Markdown",
                        data=result,
                        file_name="generated_markdown.md",
                        mime="text/markdown",
                    )
                except Exception as e:
                    st.error("Generation failed.")
                    st.exception(e)
else:
    st.info("Upload an image to generate Markdown.")
