# 🌸 Zero-Shot Flower Classification with CLIP

This project uses OpenAI's [CLIP model](https://openai.com/research/clip) to perform **zero-shot image classification** on a flower dataset. By using natural language prompts like "a blooming rose" or "a bouquet of tulips," the model attempts to match images to labels *without any additional training*.

> This project was completed as part of the FLGF24 curriculum on applied AI/ML.

---

## 🧠 What is Zero-Shot Classification?

Zero-shot classification allows models to make predictions about classes they’ve never seen during training. Instead of training a custom model, CLIP matches images to **text descriptions** using embeddings.

---

## 🔍 How It Works

1. Load the CLIP model and processor from Hugging Face `transformers`.
2. Define natural language categories.
3. Pass in flower images and compute similarity scores.
4. Rank the predictions based on confidence.

Example prompt set:

```python
["A blooming rose", "A bright daisy in the sun", "A bouquet of tulips"]
```

---

## 🌼 Example Results

| Image         | Predicted Label            | Confidence |
|---------------|----------------------------|------------|
| `daisy.jpg`   | A bright daisy in the sun  | 97.79%     |
| `iris.jpg`    | A blooming rose            | 93.07%     |
| `snowdrop.jpg`| A bouquet of tulips        | 77.28%     |

---

## 📦 Tech Stack

- Python
- `transformers` (Hugging Face)
- `torch`
- `PIL` (Python Imaging Library)
- [Kaggle Flower Dataset](https://www.kaggle.com/datasets/alsaniipe/flowerdatasets)

---

## 📁 Project Structure

```
├── flower_dataset/            # Downloaded from Kaggle
│   ├── rose.jpg
│   ├── daisy.jpg
│   ├── tulip.jpg
│   └── ...
├── zero_shot_clip.py          # Python script to run inference
├── summary.md                 # Reflection on results
├── README.md
```

---

## ✍️ Reflection

CLIP showed strong performance on images with clear matches to the prompt set, such as daisies. However, broad categories and limited phrasing resulted in mismatches for flowers like irises and snowdrops. More descriptive, tailored prompts and additional categories would likely improve classification accuracy.

This project was a great introduction to applying zero-shot learning using large vision-language models. It reinforced how much prompt engineering can influence results even without additional training.

---

## 👩‍💻 Author

Pamela Augustine  
