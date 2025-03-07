from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import kagglehub
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "faiss_db_with_metadata")
user_preferences_directory = os.path.join(db_dir, "user_preferences_db")

# Download the dataset from Kaggle
path = kagglehub.dataset_download("aadyasingh55/cocktails")
print("Path to dataset files:", path)

# Check if the FAISS vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    csv_path = os.path.join(path, "final_cocktails.csv")
    df = pd.read_csv(csv_path)

    # Prepare documents for the vector store
    documents = []
    for _, row in df.iterrows():
        content = f"""
            Name: {row.get('name', 'N/A')}
            Alcoholic: {row.get('alcoholic', 'N/A')}
            Category: {row.get('category', 'N/A')}
            Glass Type: {row.get('glassType', 'N/A')}
            Instructions: {row.get('instructions', 'N/A')}
            Ingredients: {row.get('ingredients', 'N/A')}
            Ingredient Measures: {row.get('ingredientMeasures', 'N/A')}
            """
        documents.append(Document(page_content=content, metadata={"source": row.get('id', 'N/A')}))
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
    # Load the existing vector store with the embedding function
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(persistent_directory, embeddings, allow_dangerous_deserialization=True)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 30},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model='gpt-4o-mini')

# Contextualize question prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
qa_system_prompt = """
You are a virtual bartender, named Barley, an expert in mixology and cocktail recommendations. Your tone should be warm, engaging, and conversational—like a friendly bartender chatting with a guest. Keep responses concise but informative, incorporating fun, casual language and occasional humor. Use emojis and stickers where appropriate to enhance the experience.

You must use the following pieces of retrieved context to answer the question. Extract as much relevant detail as possible from the provided context to ensure accuracy. If the user just shares his/her preferences in cocktails/ingredients but doesn`t ask you about recommendations directly (doesn`t ask you any questions) you should answer: "Okay, got it! I`ll be glad to recommend you drinks based on your preferences". Don`t give unnecessary recommendations if the user doesn`t ask! 

Also remember that apples are not pineapples, if user mentioned that he/she likes just apples(there wasn`t any mention of pineapples) that doesn`t mean he/she likes pineapples. Same with pineapples. If she/he likes pineapples that doesn`t mean she/he likes apples.

Key behaviors:
- Engage the user naturally—respond as if you're behind the bar, making recommendations and sharing insights.
- Be playful and charming—use friendly banter, light humor, and thematic expressions (e.g., ‘Shaken or stirred?’, ‘Great choice, my friend!’).
- Personalize suggestions—ask about user preferences when relevant and adapt recommendations accordingly.
- Use emoji sparingly but effectively—enhance responses with 🍹🍋🍒 when appropriate, but don’t overdo it.
- Maintain clarity and accuracy—ensure drink recipes and recommendations are correct while keeping descriptions lively.

Keep interactions fun, engaging, and immersive—just like a great bartender would! 🍸

{context}

User Preferences:
{user_preferences}
"""

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to detect user preferences using LLM
def detect_preferences_with_llm(user_message):
    prompt = f"""
    Extract the user's favorite cocktails and ingredients from the following text:

    User Message:
    "{user_message}"

    Instructions:
    1. Identify and extract any explicitly mentioned favorite cocktails (e.g., "I love Mojitos" → Mojito).
    2. Identify and extract any liked things. It can be even sugar (e.g., "I like apple and pineapple" → apple, pineapple).
    3. Identify and extract any preferred ingredients, even if stated in contrast to disliked ones (e.g., "I don’t like whiskey, but I like vodka" → vodka).
    4. Consider ingredients mentioned in a positive context (e.g., "I enjoy drinks with pineapple" → pineapple).
    5. If the user is only asking for recommendations or not stating preferences, return an empty list.
    6. Format the output as a single comma-separated list of cocktails and ingredients, without additional text or explanations.
    7. I command you to detect if the user likes apple/pineapple and other fruits.

    Example Outputs:
    - Input: "I enjoy Margaritas and anything with tequila and lime."
      Output: "Margarita, tequila, lime"
    - Input: "I love Old Fashioneds and whiskey-based drinks."
      Output: "Old Fashioned, whiskey"
    - Input: "What cocktails do you recommend?"
      Output: "[]"

    Return only the comma-separated list, with no extra words or formatting.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI assistant that detects user preferences."},
                  {"role": "user", "content": prompt}]
    )

    # Parse the response into a list of preferences
    preferences = response.choices[0].message.content.strip()
    print(preferences)

    # Handle cases where the LLM returns "[]" or invalid data
    if preferences == "[]" or not preferences:
        return []  # Return an empty list if no preferences are detected
    else:
        # Split the comma-separated string into a list of preferences
        return [pref.strip() for pref in preferences.split(",")]


# Function to store user preferences
def store_user_preferences(query, user_preferences_db):
    # Create a document from the user's query
    document = Document(page_content=query, metadata={"type": "user_preference"})

    # Add the document to the user preferences vector store
    user_preferences_db.add_documents([document])
    user_preferences_db.save_local(user_preferences_directory)

# Function to retrieve user preferences
def retrieve_user_preferences(user_preferences_db, query):
    # Retrieve relevant preferences based on the query
    relevant_preferences = user_preferences_db.similarity_search(query, k=30)
    return "\n".join([doc.page_content for doc in relevant_preferences])

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ChatRequest(BaseModel):
    message: str
    chat_history: list = []

# Define response model
class ChatResponse(BaseModel):
    response: str

# Load or create user preferences vector store
if os.path.exists(user_preferences_directory):
    user_preferences_db = FAISS.load_local(user_preferences_directory, embeddings, allow_dangerous_deserialization=True)
else:
    # Initialize with a dummy document if no user preferences exist
    dummy_document = Document(page_content="Initial dummy document", metadata={"type": "dummy"})
    user_preferences_db = FAISS.from_documents([dummy_document], embeddings)


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    query = chat_request.message
    chat_history = chat_request.chat_history

    # Detect if the user is sharing preferences using LLM
    preferences = detect_preferences_with_llm(query)

    # Only store preferences if the LLM detects valid preferences
    if preferences:  # Check if the list is not empty
        print(f"Detected preferences: {preferences}")  # Log the detected preferences
        store_user_preferences(query, user_preferences_db)
        print(f"Preferences saved to database: {preferences}")
    else:
        print("No preferences detected. Skipping database update.")

    # Retrieve relevant documents
    retrieved_docs = history_aware_retriever.invoke({"input": query, "chat_history": chat_history})

    # Retrieve user preferences
    user_preferences = retrieve_user_preferences(user_preferences_db, query)

    # Process the query using the RAG chain with user preferences
    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history,
        "context": retrieved_docs,
        "user_preferences": user_preferences
    })

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result["answer"]))

    return ChatResponse(response=result["answer"])

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)