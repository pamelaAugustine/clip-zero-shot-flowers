from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import random

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to flower dataset
dataset_path = "./flower_dataset/flowers"

# Custom categories
categories = ["A blooming rose", "A bright daisy in the sun", "A bouquet of tulips"]

# Gather all image paths from the subfolders in 'flowers'
all_images = []
for subdir, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):  # Ensure image files are selected
            all_images.append(os.path.join(subdir, file))

# Select 5 random images from the dataset
test_images = random.sample(all_images, 5)

# Perform classification for each selected image
for image_path in test_images:
    image = Image.open(image_path)
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Display results
    print(f"\nResults for {os.path.basename(image_path)} (from {os.path.dirname(image_path).split('/')[-1]}):")
    for category, prob in zip(categories, probs[0]):
        print(f"  {category}: {prob:.2%}")
