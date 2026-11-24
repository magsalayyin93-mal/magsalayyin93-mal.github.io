import io

import gradio as gr
import torch
from PIL import Image, ImageChops
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "zodumair/document-forgery-detector"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()


def prepare_ela(image: Image.Image) -> Image.Image:
    original = image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    ela = ImageChops.difference(original, recompressed)
    max_diff = max(channel[1] for channel in ela.getextrema()) or 1
    ela = ela.point(lambda pixel: min(255, int(pixel * (255.0 / max_diff) * 1.5)))
    return Image.blend(original, ela, alpha=0.3)


def analyze_document(image: Image.Image):
    if image is None:
        return {"available": False, "verdict": "Needs Further Verification", "message": "Upload a document image first."}

    prepared = prepare_ela(image)
    inputs = processor(images=prepared, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    index = int(torch.argmax(probabilities).item())
    label = str(model.config.id2label.get(index, "unknown"))
    confidence = float(probabilities[index].item())
    normalized = label.casefold()

    if any(word in normalized for word in ("forged", "fake", "tampered", "fraud")):
        model_class = "suspicious"
    elif any(word in normalized for word in ("real", "authentic", "genuine")):
        model_class = "genuine"
    else:
        model_class = "unknown"

    if confidence < 0.70 or model_class == "unknown":
        verdict = "Needs Further Verification"
        message = "The model result is uncertain. Review the original document and verify it with the issuer."
    elif model_class == "suspicious":
        verdict = "Suspicious"
        message = "The model found visual patterns associated with forged documents. This is a screening flag, not proof of fraud."
    else:
        verdict = "Potentially Genuine"
        message = "The model classified this as potentially genuine. This is not proof of authenticity; review the original document."

    return {
        "available": True,
        "verdict": verdict,
        "label": label,
        "confidence": round(confidence, 4),
        "message": message,
        "model": MODEL_ID,
    }


with gr.Blocks(title="DocuVerify Tampering Model") as demo:
    gr.Markdown(
        "# DocuVerify Visual Screening\n"
        "Upload a JPG, JPEG, or PNG receipt or invoice for preliminary document-forgery screening. "
        "The result is not definitive verification."
    )
    image = gr.Image(type="pil", label="Receipt or invoice")
    result = gr.JSON(label="Model result")
    button = gr.Button("Analyze document", variant="primary")
    button.click(analyze_document, inputs=image, outputs=result, api_name="predict")

if __name__ == "__main__":
    demo.launch()
