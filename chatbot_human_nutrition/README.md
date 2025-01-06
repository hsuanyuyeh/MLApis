## Chatbot with rag mechanism
Build a chatbot with rag mechanism to answer nutrition related question. UI was built in template with pdf uploading feature.

#### Steps
1. Document preprocessing
2. Create the chromadb with uploaded pdf file.
3. Launch the choosed LLM model with a retriever to chromadb
4. Launch the chatbot with LLM in the backend and keep track the conversation history
\
\
![screenshot](https://github.com/hsuanyuyeh/MLApis/blob/main/chatbot_human_nutrition/rag_bot_app.png)

#### Document preprocessing - preparing files to generate the best answers
1. Document loaders - langchain_community.document_loaders: pdf, txt, markdown, json, csv, webbase...
2. Text splitter - langchain.text_splitter: CharacterTextSplitter, RecursiveCharacterTextSplitter MarkdownHeaderTextSplitter...
3. Embedding models - langchain_community.embeddings: HuggingFaceEmbeddings
4. Vector store DB - langchain_community.vectorstores: Chroma, FIASS
5. Retriever - simple similarity search, maximum marginal relevance search (MMR), multi-query retriever, self-query retriever, parent document retriever...

