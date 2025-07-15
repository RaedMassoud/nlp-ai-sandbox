from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "I can not log into my account and need help"
labels = ["billing", "technical issue", "account access", "feedback"]

result = classifier(text, candidate_labels=labels)

print("Prediction: ", result["labels"][0])