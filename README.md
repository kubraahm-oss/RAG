# RAG Project Report

## 1. Project Overview
This project builds a simple Retrieval-Augmented Generation (RAG) system using two documents:
- Employee_Rights_Charter.pdf
- Company_Policy_Manual.docx

The goal was to create a small end-to-end workflow that can load documents, split them into searchable chunks, create embeddings, store them in a FAISS index, and answer questions using only the retrieved context.

## 2. What Was Done

### Step 1: Environment setup
The notebook installs the required libraries for document loading, chunking, vector storage, and model inference.

```python
!pip install -U langchain langchain-huggingface langchain-community langchain-classic faiss-cpu tiktoken docx2txt pypdf sentence-transformers
```

### Step 2: Load the source documents
The PDF and DOCX files were loaded into the notebook and combined into one document list.

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

pdf_path = Path("Employee_Rights_Charter.pdf")
docx_path = Path("Company_Policy_Manual.docx")

pdf_docs = PyPDFLoader(str(pdf_path)).load()
docx_docs = Docx2txtLoader(str(docx_path)).load()

all_docs = pdf_docs + docx_docs
```

### Step 3: Split the content into chunks
The document text was split into smaller chunks to improve retrieval quality.

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(all_docs)
```

### Step 4: Create embeddings and build the FAISS index
Embeddings were generated from the chunks and stored in a vector database for similarity search.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### Step 5: Build the retrieval QA chain
A retriever was created from the FAISS index and connected to a language model and prompt template.

```python
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

qa_prompt = PromptTemplate.from_template(
    """You are a precise HR policy assistant. Use only the provided context.
If the answer is not in context, say: I don't know based on the provided documents.
Return one short, direct answer sentence only.

Context:
{context}

Question: {question}
Answer:"""
)
```

### Step 6: Ask a question
The final step was to send a question to the QA chain and receive an answer based on the retrieved documents.

```python
query = "give me 3 bullet points of the company policies that which is unfair to employees and should be changed"
result = qa_chain.invoke({"query": query})
print(result["result"])
```

## 3. Example Output
Typical output from the notebook includes messages such as:

```text
Loaded pages/docs: PDF=10, DOCX=8, TOTAL=18
Created chunks: 18
FAISS index saved to ./faiss_index
Stored chunks: 18
Using hosted Hugging Face inference.
RAG chain is ready. Use qa_chain.invoke({'query': 'your question'})
```

A sample answer may look like:

```text
The policy appears to limit employee rights in a way that may be unfair, especially if it restricts employees from raising concerns or requesting support.
```

## 4. Challenges Encountered

### Challenge 1: Environment dependency
The project depends on several packages and model downloads. If one package is missing or the environment is not set up correctly, the notebook may fail early.

### Challenge 2: Model availability
The language model may not always work with hosted inference. In that case, the notebook falls back to a local model, which can be slower or produce less reliable answers.

### Challenge 3: Token and access issues
A Hugging Face token is helpful for hosted inference. Without it, the workflow may need to rely on local inference or may produce weaker outputs.

### Challenge 4: Retrieval quality
The answer quality depends strongly on how well the relevant context is retrieved. If the chunking strategy or similarity settings are not tuned well, the model may return incomplete or less accurate answers.

## 5. Key Learnings
- RAG works best when documents are split into useful chunks.
- Vector search is essential for finding relevant context efficiently.
- Prompt design strongly affects the quality of the answer.
- The system is only as good as the input documents and retrieval quality.

## 6. Project Files
- rag_two_files.ipynb: Main notebook containing the full workflow
- faiss_index/: Saved FAISS vector store
- Employee_Rights_Charter.pdf: Source PDF document
- Company_Policy_Manual.docx: Source DOCX document
- README.md: Project documentation and report

## 7. Results and Conclusion
The project was successful in building a working prototype of a document-based question-answering system. The notebook successfully loaded both documents, created embeddings, stored them in a FAISS index, and generated answers using a retrieval-based workflow.

### Results achieved
- Two different document formats were processed successfully: PDF and DOCX.
- The system was able to split the content into retrievable chunks.
- A vector store was created and saved locally for reuse.
- The QA chain returned context-based answers to user questions.

### Conclusion
This implementation demonstrates the core idea of RAG clearly: retrieve relevant information from documents first, then use a language model to generate a response grounded in that retrieved context. Although the system is still a basic prototype, it provides a strong foundation for more advanced applications such as larger document collections, better chunking strategies, improved prompts, and more accurate retrieval models.
