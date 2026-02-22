# 📊 Análise de Sentimentos com BERT + SVM

Este projeto apresenta um experimento de **classificação de sentimentos em textos de redes sociais**, utilizando embeddings extraídos do modelo BERT e um classificador Support Vector Machine (SVM).

A proposta é demonstrar como modelos Transformer pré-treinados podem ser utilizados como extratores de características semânticas, sendo posteriormente integrados a algoritmos clássicos de Machine Learning.

---

## 🎯 Objetivo

Classificar textos de redes sociais em diferentes categorias de sentimento utilizando:

- Representações contextuais geradas pelo BERT
- Classificação supervisionada com SVM
- Visualização da separação das classes via PCA

---

## 🧠 Pipeline Experimental

O fluxo do experimento segue as seguintes etapas:

### 1️⃣ Carregamento do Dataset
- Dataset de análise de sentimentos em redes sociais
- Download automatizado via `kagglehub`
- Leitura com Pandas

### 2️⃣ Extração de Embeddings com BERT
- Tokenização com `bert-base-uncased`
- Processamento com `BertModel`
- Extração do embedding do token `[CLS]`
- Conversão para vetores NumPy

### 3️⃣ Divisão dos Dados
- Separação treino/teste com `train_test_split`

### 4️⃣ Treinamento do Modelo
- Classificador **SVM com kernel linear**
- Treinamento supervisionado

### 5️⃣ Avaliação
- Cálculo de acurácia
- Relatório completo de classificação (precision, recall, f1-score)

### 6️⃣ Visualização
- Redução de dimensionalidade com **PCA**
- Plotagem bidimensional dos embeddings
- Visualização da separação entre classes

---

## 🤖 Abordagem Metodológica

O projeto utiliza uma abordagem híbrida:

- **BERT como extrator semântico profundo**
- **SVM como classificador linear supervisionado**

Essa estratégia permite:
- Capturar relações contextuais complexas do texto
- Manter um modelo de classificação eficiente e interpretável
- Visualizar a organização dos dados no espaço vetorial

---

## 🛠 Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Matplotlib
- KaggleHub

---

## 📊 Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-score
- Visualização com PCA

---

## 🚀 Como Executar

### 1️⃣ Instale as dependências

```bash
pip install kagglehub[pandas-datasets]
pip install sentence-transformers
pip install transformers torch scikit-learn matplotlib pandas numpy
```

###2️⃣ Execute o script ou notebook

O código pode ser executado em:

Google Colab
Jupyter Notebook
Ambiente local com Python 3
