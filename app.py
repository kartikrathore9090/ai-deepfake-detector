from deepfake_detector import highlight_face
from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")

@app.route("/", methods=["GET","POST"])
def index():

    result = None
    confidence = None
    output_image = None

    if request.method == "POST":

        file = request.files["image"]
        image = Image.open(file)

        image_path = "static/input.jpg"
        image.save(image_path)
        
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        predicted = torch.argmax(probs)
        confidence = probs[0][predicted].item()

        if predicted == 0:
            result = "REAL IMAGE"
        else:
            result = "DEEPFAKE IMAGE"
            
        output_image = highlight_face(image_path)

    return render_template(
    "index.html",
    result=result,
    confidence=confidence,
    output_image=output_image
)

if __name__ == "__main__":
    app.run(debug=True)