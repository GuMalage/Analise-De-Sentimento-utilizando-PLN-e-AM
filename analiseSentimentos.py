#Execute no terminal: 
#!pip install kagglehub[pandas-datasets] -q
#!pip install sentence-transformers --quiet

import kagglehub
import pandas as pd
import os
from transformers import BertTokenizer, BertModelo
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = kagglehub.dataset_download(
    "kashishparmar02/social-media-sentiments-analysis-dataset"
)

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)
text_col = "Text"
label_col = "Sentiment"
texts = df[text_col].astype(str).tolist()
labels = df[label_col].astype(str).tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(
    texts[:52],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
labels = np.array(labels[:52])

X_train, X_test, y_train, y_test = train_test_split(
    cls_embeddings,
    labels,
    test_size=0.2,
    random_state=42
)


svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
plt.figure(figsize=(8,6))
unique_labels = np.unique(y_train)

for label in unique_labels:
    mask = y_train == label
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=label,
        s=30
    )

plt.title("Distribuição das amostras (PCA dos embeddings)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Sentimento")
plt.show()

print("Acurácia:", accuracy_score(y_test, y_pred))

print("\nRelatório de Classificação:\n")
print(classification_report(y_test, y_pred))