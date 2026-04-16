from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


class ChatWithPDF:
    def __init__(self, pdf_path):
        loader = self.load_pdf(pdf_path=pdf_path)
        split_docs = self.split_documents(documents=loader)
        embeds = self.embedder()
        self.vector_store = self.my_vector_store(doc=split_docs, embeds=embeds)
        self.model = self.ai_model()
        self.agent = self.my_agent()    


    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350)
        split_documents = splitter.split_documents(documents)
        return split_documents
    
    def embedder(self):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        return embeddings
    
    def my_vector_store(self, doc, embeds):
        vector_store = Chroma.from_documents(documents=doc, embedding=embeds)
        return vector_store

    def ai_model(self):
        model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2048,
        )
        return model
    
    def my_proposed_tools(self):
        @tool(description="This tool retrieve   the chunks of documents based on the query")
        def retrieve_chunks(query: str):
            """Retrieve the chunks of documents based on the query"""   
            results = self.vector_store.similarity_search(query)
            serilezed = '\n\n'.join([doc.page_content for doc in results])
            return serilezed
        return [retrieve_chunks]

    def my_agent(self):
        tools = self.my_proposed_tools()
        agent = create_agent(
            model=self.model,
            tools=tools,
            checkpointer=InMemorySaver(),
            middleware = [SummarizationMiddleware(model=ChatGroq(model="llama-3.1-8b-instant"), trigger=("tokens", 1500), keep=("messages", 10))],
            system_prompt="You are a helpful assistant that answers questions based on the provided documents. otherwise replay, I dont know about it.",
        )       
        return agent
    
    def ask_question(self, query, thread_id):
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.agent.stream({"messages": [{"role": "user", "content": query}]},config=config, stream_mode="values"):
            if "messages" not in chunk:
                continue
            msg = chunk["messages"][-1]
            # sirf AIMessage handle karo
            if msg.__class__.__name__ == "AIMessage":
                content = msg.content
                # content list hota hai (Gemini style)
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            yield item["text"]
                # fallback (agar string ho)
                elif isinstance(content, str):
                    yield content

# bot = ChatWithPDF(pdf_path=r"C:\Users\anwaa\Downloads\llm_response.pdf")
# while True:
#     query = input("You: ")
#     if query == "exit":
#         break 
#     print("bot: ", end="") 
#     for b in bot.ask_question(query):
#         print(b, end="\n", flush=True)
     
