import json
import os
import uuid
from deepfake_detector import highlight_face
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)

# Load Model
processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    output_image = None
    filename = None 

    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)

        upload_folder = "static"
        os.makedirs(upload_folder, exist_ok=True)

        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(upload_folder, filename)
        image.save(image_path)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        predicted = torch.argmax(probs)
        confidence = probs[0][predicted].item()

        # Model mapping: check if 0 is Real or Fake for your specific model
        if predicted == 0:
            result = "REAL IMAGE"
        else:
            result = "DEEPFAKE IMAGE"

        # Highlight suspicious region
        output_image = highlight_face(image_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        output_image=output_image,
        uploaded_image=filename
    )

@app.route("/feedback", methods=["POST"])
def feedback():
    # IMPORTANT: Use the exact 'name' attributes from your HTML form
    name = request.form.get("name")
    rating = request.form.get("rating")
    message = request.form.get("message")
    # This captures the 'yes' or 'no' from your hidden input
    prediction_correct = request.form.get("prediction_correct") 

    data = {
        "name": name,
        "rating": rating,
        "message": message,
        "was_prediction_correct": prediction_correct
    }

    file_path = "feedback.json"

    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = []
    else:
        feedback_data = []

    # Add new entry
    feedback_data.append(data)

    # Save back to file
    with open(file_path, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
