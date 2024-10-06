# **Adaptive RAG Query Routing System**
https://files.oaiusercontent.com/file-BvzBPOBjLRDEzrO95VqMb7Hk?se=2024-10-06T06%3A27%3A01Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D299%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3Dimage.png&sig=MvY3wDIWW71NLxG%2Bcu0sBqaReSbW9Ef6x2oLXpp9iEg%3D![image](https://github.com/user-attachments/assets/836d7d1d-cad1-4a13-8d8a-eb1e54b31e40)



The **Adaptive Retrieval-Augmented Generation (RAG) Query Routing System** intelligently routes user queries to the most relevant data source. By integrating multiple **Large Language Models (LLMs)** such as **Fireworks**, **GPT**, and **Gemini** with **LangChain** and **Chroma DB**, the system optimizes query routing based on query type, improving efficiency. Additionally, it utilizes **LangSmith** and **LangGraph** for workflow visualization and error tracing, which helps reduce operational downtime.

## **Key Features**
- **Intelligent Query Routing**: Automatically routes queries to either a vectorstore or web search based on their nature.
- **Multiple LLM Integration**: Incorporates **Fireworks**, **GPT**, and **Gemini** for enhanced response generation.
- **Query Complexity Classification**: Differentiates between simple and complex queries, generating tailored responses.
- **LangSmith & LangGraph**: Workflow visualization and error tracing tools improve system performance and reduce downtime.
- **Optimized Resource Allocation**: Efficiently manages system resources, reducing operational costs.

## **System Flow**
1. **Start**: The system receives a query.
2. **Web Search**: If identified as a general query, it performs a web search and generates a response.
3. **Retrieve from Vectorstore**: For domain-specific queries, the system retrieves information from the vectorstore.
4. **Determine Question Type**: The system classifies the question as either **Simple** or **Complex**.
5. **Generate Response**: Depending on the classification, it generates either a simple or complex response.

## **Installation**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/adaptive-rag-query-routing-system.git
    cd adaptive-rag-query-routing-system
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the system**:
    ```bash
    python main.py
    ```


## **Contact**

- **Author**: Ankit U Patil
- **GitHub**: [AnKiTu03](https://github.com/AnKiTu03)
- **Email**: ankitupatil1@gmail.com

---
