import streamlit as st
from transformers import ViltForQuestionAnswering, BlipForQuestionAnswering, AutoProcessor
from PIL import Image
from modules.model import CustomViltForVQA
from modules.beit_3 import Beit3Processing
from timm.models import create_model
import modeling_finetune
from collections import OrderedDict
import torch

models = {
    "BLIP": (AutoProcessor, BlipForQuestionAnswering, "Salesforce/blip-vqa-base"),
    "ViLT": (AutoProcessor, ViltForQuestionAnswering, "dandelin/vilt-b32-finetuned-vqa"),
    "My Model": (AutoProcessor, CustomViltForVQA, "phonghoccode/vilt-vqa-finetune-pytorch")
}

def get_format_response(image,question,selected_model):
    if selected_model == 'beit3':
        processor = Beit3Processing()
        model = create_model(
            "beit3_base_patch16_480_vqav2_vqav2",
            pretrained=False,
            drop_path_rate=0.1,
            vocab_size=64010,
            checkpoint_activations=None,
            num_classes=108
        )

        # Load checkpoint
        checkpoint = torch.load("/home/phongcoder/Workspace/TextMining-VQA/beit3/output/checkpoint-best/mp_rank_00_model_states.pt", map_location=device)
        if "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
            checkpoint = checkpoint["model"]
        elif "module" in checkpoint.keys():
            model.load_state_dict(checkpoint["module"])
            checkpoint = checkpoint["module"]
        else:
            new_state_dict = OrderedDict()

            for key, value in checkpoint.items():
                new_key = key.replace("module.", "")  # Remove "module."
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict)
        data = processor(image, question)
        logits = model(
            image=data["image"], 
            question=data["language_tokens"],
            padding_mask=data["padding_mask"]
        )
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer

    processor, model_class, model_name = models[selected_model]
    processor = processor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    
    encoding = processor(image, question, return_tensors="pt")
    
    if selected_model in ['ViLT', 'My Model']:
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer
    else:
        outputs = model.generate(**encoding)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return answer

def run():
    st.title("Visual Question Answering (VQA)")
    st.subheader("A demo app showcasing VQA models.")

    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image")

    question = st.text_input("Ask a Question about the Image")

    if uploaded_image and question:
        answer = get_format_response(image, question, selected_model)
        st.write(f"🤔 {selected_model} Answer: {answer}")

run()
