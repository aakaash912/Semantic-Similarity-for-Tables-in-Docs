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
2. Install Dependencies within the code

Usage
Step 1: Prepare Your Word Document
Ensure that your Word document contains tables. The app will ignore non-tabular content.
Save your Word document (e.g., tables.docx).

Step 2: Run the Application (Ensure to include the path of your Word Document for Parsing)

The app will extract tables, generate embeddings, and display matching tables side-by-side with similarity scores.

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
Include the Option to upload files to Gradio.
Add an option to compare models side-by-side.
Save results to a CSV file for further analysis.
Support for additional models via Hugging Face.
