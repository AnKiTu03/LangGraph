Adaptive RAG Query Routing System

The Adaptive Retrieval-Augmented Generation (RAG) Query Routing System is designed to intelligently route user queries to the most relevant data source by integrating multiple Large Language Models (LLMs) such as Fireworks, GPT, and Gemini with LangChain and Chroma DB. It optimizes query routing, determining whether to access a vectorstore or perform a web search, depending on the query type. This system improves efficiency by using LangSmith and LangGraph for workflow visualization and error tracing, reducing operational downtime.

Key Features

Intelligent Query Routing: Routes user queries to a vectorstore or web search based on the query type.
Multiple LLMs: Integrates Fireworks, GPT, and Gemini to provide optimized query responses.
Query Classification: Classifies queries as either simple or complex, generating appropriate responses.
LangSmith & LangGraph: Visualizes workflows and traces errors, improving system performance and reducing downtime.
Optimized Resource Allocation: Reduces operational costs by efficiently managing resources.
System Flow

Web Search: Queries identified as web searches are routed to generate responses based on external search results.
Retrieve from Vectorstore: Queries that require domain-specific or internal knowledge are routed to retrieve data from the vectorstore.
Simple or Complex Responses: The system determines if the response is simple or complex and generates appropriate answers.
