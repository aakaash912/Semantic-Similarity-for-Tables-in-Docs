# Semantic-Similarity-for-Tables-in-Docs
Finding out best matching pairs of Tables based on their contents' semantic similarities using Cosine Similarity
This application compares semantic similarity between tables extracted from a Word document. The app evaluates the similarity using two models:

Ollama LLaMA 3.1
all-MiniLM-L6-v2 (SentenceTransformer)
The comparison is displayed through a Gradio interface, showing the best matching tables and their similarity scores.

Features
Extracts tables from a Word document.
Generates embeddings for tables using:
Ollama’s LLaMA 3.1
Hugging Face’s all-MiniLM-L6-v2 model.
Performs cosine similarity on the embeddings.
Displays matching tables side-by-side in a Gradio UI with similarity scores.
Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/table-semantic-similarity.git
cd table-semantic-similarity
2. Install Dependencies
Ensure that Python 3.8+ is installed. Then, run:

bash
Copy code
pip install -r requirements.txt
requirements.txt should include:

text
Copy code
gradio
sentence-transformers
langchain
ollama
python-docx
scikit-learn
Usage
Step 1: Prepare Your Word Document
Ensure that your Word document contains tables. The app will ignore non-tabular content.
Save your Word document (e.g., tables.docx).
Step 2: Run the Application
bash
Copy code
python app.py
Step 3: Upload Your Word Document
Use the Gradio interface to upload the Word document.
The app will extract tables, generate embeddings, and display matching tables side-by-side with similarity scores.
Code Overview
1. Parse Tables from Word Document
python
Copy code
from docx import Document

def extract_tables(docx_path):
    doc = Document(docx_path)
    tables = []

    for table in doc.tables:
        table_content = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        tables.append(table_content)

    return tables
2. Generate Embeddings with Two Models
Ollama LLaMA 3.1 Embeddings:
python
Copy code
from langchain.embeddings import OllamaEmbeddings

llama_model = OllamaEmbeddings(model="llama3.1")
llama_embeddings = [llama_model.embed_documents(["\n".join(row) for row in table]) for table in tables]
SentenceTransformer all-MiniLM-L6-v2 Embeddings:
python
Copy code
from sentence_transformers import SentenceTransformer

mini_lm_model = SentenceTransformer('all-MiniLM-L6-v2')
mini_lm_embeddings = mini_lm_model.encode(["\n".join([" | ".join(row) for row in table]) for table in tables], convert_to_tensor=True)
3. Calculate Cosine Similarity
python
Copy code
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

llama_similarity = get_similarity_matrix(llama_embeddings)
mini_lm_similarity = get_similarity_matrix(mini_lm_embeddings)
4. Display Results with Gradio
python
Copy code
import gradio as gr

def display_results(similarity_matrix, tables):
    sorted_pairs = get_sorted_similarities(similarity_matrix)
    content = ""

    for (i, j), score in sorted_pairs:
        content += f"<h3>Table {i} <--> Table {j} | Similarity: {score:.4f}</h3>"
        content += f"<div style='display: flex; gap: 20px;'>"
        content += convert_table_to_html(tables[i - 1])
        content += convert_table_to_html(tables[j - 1])
        content += "</div><hr>"

    return content

interface = gr.Interface(
    fn=lambda: display_results(llama_similarity, tables),
    inputs=[],
    outputs="html",
    title="Table Semantic Similarity with LLaMA 3.1 vs. MiniLM",
)

interface.launch()
How It Works
Upload: A DOCX file containing tables.
Embedding Generation:
Tables are converted into embeddings using both LLaMA 3.1 and all-MiniLM-L6-v2.
Similarity Calculation:
Cosine similarity is computed between table embeddings.
Display: The app renders matching tables side-by-side along with similarity scores in a Gradio interface.
Expected Output
Gradio interface displays matching tables side-by-side, like:
less
Copy code
Table 1 <--> Table 3 | Similarity: 0.8503
[Table 1] [Table 3]

Table 2 <--> Table 4 | Similarity: 0.7921
[Table 2] [Table 4]
To-Do / Future Improvements
Add an option to compare models side-by-side.
Save results to a CSV file for further analysis.
Support for additional models via Hugging Face.
