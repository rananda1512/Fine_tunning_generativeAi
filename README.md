
# 🌿 **Plant Disease Question-Answering Model** 🌿

**Plant Disease Question-Answering Model** is an AI-driven solution aimed at answering plant disease-related queries using the power of **Flan-T5** model. This project fine-tunes a pre-trained transformer model on a custom dataset to help anyone in agriculture, botany, or plant care easily find answers to common plant diseases.

---

## 🚀 **Getting Started**

### 1. **Install the Required Libraries**

First, make sure to install the required dependencies for the project by running the following command in your terminal:

```bash
pip install datasets transformers transformers[torch] torch huggingface-hub nltk rouge_score evaluate pandas
```

These libraries include the tools for model training, evaluation, and dataset handling.

---

## 📝 **Overview of the Project**

The main objective of this project is to train a model capable of answering questions about plant diseases. Here's the workflow:

1. **Dataset**: We use a custom dataset of plant diseases and their descriptions.
2. **Preprocessing**: Each question is prepended with a prefix, and both the questions and answers are tokenized.
3. **Model**: The model used is the **google/flan-t5-base** model, fine-tuned for question-answering tasks.
4. **Training**: The model is trained on this custom dataset, and we evaluate its performance using **ROUGE** metrics.
5. **Inference**: Once the model is trained, we can use it to answer any plant disease-related questions.

---

## 🔧 **How the Code Works**

### 1. **Dataset Creation and Saving**

We begin by preparing the dataset, consisting of questions and their corresponding answers related to plant diseases. The data is saved in a CSV format.

```python
import pandas as pd

plant_disease_data = [
    {"question": "What are Rust for plant?", "answer": "Rust refers to a condition in plants caused by fungal diseases from the order Pucciniales..."},
    {"question": "What causes yellowing and wilting of leaves on rice plants?", "answer": "This disease is caused by the bacterial pathogen Xanthomonas oryzae."},
    ...
]

df = pd.DataFrame(plant_disease_data)
df.to_csv('plant_disease_data.csv', index=False, encoding='utf-8')
print("Dataset successfully saved in CSV format!")
```

---

### 2. **Loading the Dataset and Splitting It**

After saving the dataset, we load it and split it into training and test sets. The training set will be used for model fine-tuning, while the test set will be used to evaluate its performance.

```python
from datasets import load_dataset

dataset = load_dataset('csv', data_files='plant_disease_data.csv')
data = dataset['train'].train_test_split(test_size=0.3)
print(data)
```

---

### 3. **Data Preprocessing**

Before passing the data to the model, we preprocess it by adding a prefix to each question to guide the model's response.

```python
prefix = "answer this question:"

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=examples["answer"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = data.map(preprocess_function, batched=True)
```

---

### 4. **Model Setup**

We use the pre-trained **Flan-T5** model. The tokenizer and model are loaded from the Hugging Face Model Hub. Additionally, a **DataCollator** is set up for padding the inputs during training.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq

MODEL_NAME = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
```

---

### 5. **Evaluating with ROUGE Metric**

We use the **ROUGE** metric to evaluate the model's performance by comparing the generated answers with the ground truth answers.

```python
import nltk
from evaluate import load

nltk.download('punkt')

metric = load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge metric evaluation
    decoded_preds = ["
".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["
".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result
```

---

### 6. **Training the Model**

The model is trained using the **Seq2SeqTrainer** from Hugging Face's transformers library. The training parameters such as learning rate, batch size, and number of epochs are customized to optimize the model.

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
```

---

### 7. **Answering Questions with the Fine-Tuned Model**

Once training is complete, the model can be used to answer plant disease-related questions. We input a question and generate a response using the fine-tuned model.

```python
# Load the fine-tuned model
finetuned_model = T5ForConditionalGeneration.from_pretrained("./results/checkpoint-42")
tokenizer = T5Tokenizer.from_pretrained("./results/checkpoint-42")

# Example question
my_question = "What part of the plant is most affected by Powdery Mildew?"

# Tokenize input question
inputs = "answer to this question: " + my_question
inputs = tokenizer(my_question, return_tensors="pt")

# Generate answer
outputs = finetuned_model.generate(
    **inputs,
    max_length=150,
    min_length=50,
    length_penalty=0.8,
    no_repeat_ngram_size=3,
    temperature=0.7,
    top_p=0.9
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

---

## 💡 **Key Features**

- **Real-time Inference**: Ask any plant disease-related question and get an instant answer.
- **Fine-tuned Model**: Tailored specifically for plant diseases, ensuring highly accurate answers.
- **Comprehensive Dataset**: Includes a wide range of questions about plant diseases like rust, mildew, blight, and more.
- **ROUGE Evaluation**: Measures the quality of generated answers using a well-established evaluation metric.

---

## 🚀 **Potential Use Cases**

- **Agricultural Experts**: Quickly get answers about plant diseases to support better crop management.
- **Botany Enthusiasts**: Learn about plant diseases and their treatments.
- **Farmers**: Use the model for fast diagnosis of plant disease symptoms to improve crop health.

---

## 🔥 **Conclusion**

This **Plant Disease Question-Answering Model** leverages the power of deep learning and natural language processing to provide a robust solution for identifying and treating plant diseases. With the fine-tuned **Flan-T5** model, it offers a fast and accurate way to answer questions related to plant care and disease management.

---

Feel free to contribute, ask questions, or explore the model further. Happy farming! 🌱
