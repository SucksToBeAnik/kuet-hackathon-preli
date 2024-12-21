from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
import uvicorn

from sqlmodel import SQLModel, select, Session
from config.db import get_session, engine
from models.models import Ingredient
from contextlib import asynccontextmanager


# langchain related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
import faiss


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables
    SQLModel.metadata.create_all(engine)
    yield
    # Drop tables
    # SQLModel.metadata.drop_all(engine)


app = FastAPI(lifespan=lifespan)


llm = ChatOllama(base_url="http://localhost:11434", model="llama3.2:1b")

embedding_function = OllamaEmbeddings(
    base_url="http://localhost:11434", model="nomic-embed-text"
)

# Get the dimension of the embeddings
sample_text = "This is a sample text to get the embedding dimension."
sample_embedding = embedding_function.embed_query(sample_text)
embedding_dim = len(sample_embedding)

# Initialize FAISS index with the correct dimension
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/ingredients")
def get_ingredients(session: Session = Depends(get_session)):
    query = select(Ingredient)
    ingredients = session.exec(query).all()
    return ingredients


@app.post("/ingredients")
def create_ingredient(ingredient: Ingredient, session: Session = Depends(get_session)):
    session.add(ingredient)
    session.commit()
    return ingredient


# provide files
@app.post("/files")
def upload_file(file: UploadFile = File(...), session: Session = Depends(get_session)):
    # if file.content_type != "application/text":
    #     raise HTTPException(status_code=400, detail="Invalid file type")
    contents = file.file.read()
    texts = contents.decode("utf-8")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([texts])

    vector_store.add_documents(chunks)
    return {"message": "File uploaded successfully"}


@app.get("/query")
def query_vectorstore(query: str, session: Session = Depends(get_session)):
    try:
        # Get ingredients and relevant documents
        db_query = select(Ingredient)
        available_ingredients = session.exec(db_query).all()
        docs = vector_store.similarity_search(query)

        if len(docs) == 0:
            return {"message": "No relevant recipes or information found"}

        # Format ingredients list for the prompt
        ingredients_list = "\n".join([f"- {ing.name}" for ing in available_ingredients])

        # Create a more detailed prompt
        prompt = PromptTemplate(
            input_variables=["query", "context", "ingredients"],
            template="""You are a helpful chef assistant. Based on the following information and available ingredients, suggest possible recipes and cooking instructions.

Query: {query}

Available Ingredients:
{ingredients}

Relevant Cooking Information:
{context}

Please suggest recipes that:
1. Can be made primarily with the available ingredients
2. Match the user's query
3. Are practical and detailed

If the available ingredients for the recipe are not present, mention that the recipe cannot be made.
""",
        )

        # Combine document contents
        context = "\n".join([doc.page_content for doc in docs])

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke(
            {"query": query, "ingredients": ingredients_list, "context": context}
        )

        return {
            "query": query,
            "answer": answer,
            "docs": docs,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
