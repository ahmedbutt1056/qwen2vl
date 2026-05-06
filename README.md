---
title: Document Image to Markdown Generator
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Document Image to Markdown Generator

This Streamlit app uses Qwen2-VL-2B-Instruct with a QLoRA adapter trained for document image to Markdown generation.

## Adapter option 1

Put the trained adapter folder in the same repo:

final_markdown_model/

## Adapter option 2

Upload adapter to a Hugging Face model repo and set Space variable:

ADAPTER_MODEL_NAME = your-username/your-adapter-repo
