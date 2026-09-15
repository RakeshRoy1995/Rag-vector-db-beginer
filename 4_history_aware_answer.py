import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


persistent_directory = "db/chroma_db"


# --------------------------------------------------
# 1. Load your existing embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=os.environ["MODEL_NAME"]
)


# --------------------------------------------------
# 2. Load ChromaDB
# --------------------------------------------------

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# 3. Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


# --------------------------------------------------
# 4. User chat history
# --------------------------------------------------

chat_history = []


def ask_question(user_question):

    print(f"\n--- You asked: {user_question} ---")


    # --------------------------------------------------
    # Step 1: Rewrite question using conversation history
    # --------------------------------------------------

    if chat_history:

        messages = [
            SystemMessage(
                content="""
                Rewrite the user's new question into a standalone
                search query.

                Use conversation history ONLY when the new question
                depends on previous context.

                If the new question is already a clear topic,
                company, person, or entity, keep it as that topic.

                Examples:

                New question: Tesla
                Output: Tesla

                New question: Where is it headquartered?
                Output: Where is Tesla headquartered?

                Return ONLY the rewritten search query.
                """
            )
        ] + chat_history + [
            HumanMessage(
                content=f"New question: {user_question}"
            )
        ]

        result = llm.invoke(messages)

        # ChatGoogleGenerativeAI returns AIMessage
        search_question = result.content.strip()

    else:

        search_question = user_question


    print(f"Searching for: {search_question}")


    # --------------------------------------------------
    # Step 2: Retrieve documents
    # --------------------------------------------------

    retriever = db.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")


    for i, doc in enumerate(docs, 1):

        lines = doc.page_content.split("\n")[:10]

        preview = "\n".join(lines)

        print(f"\nDoc {i}:")
        print(preview)


    # --------------------------------------------------
    # Step 3: Build RAG context
    # --------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    combined_input = f"""
Question:
{user_question}


Retrieved Documents:
--------------------
{context}
--------------------


Answer the question using ONLY the retrieved documents.

Rules:

1. Do not use your own knowledge.
2. Do not use previous conversation as evidence.
3. Ignore unrelated documents.
4. Do not make assumptions.
5. If the answer is not contained in the documents, say:

"I don't have enough information to answer that question
based on the provided documents."
"""


    # --------------------------------------------------
    # Step 4: Generate answer using Gemini
    # --------------------------------------------------

    messages = [

        SystemMessage(
            content="""
            You are a RAG question-answering assistant.

            Answer ONLY using the retrieved documents.

            Rules:

            - Do not use your own knowledge.
            - Do not use previous conversation as evidence.
            - Ignore unrelated documents.
            - Do not make assumptions.
            - If the answer is not in the documents, say:

              "I don't have enough information to answer that question
              based on the provided documents."
            """
        ),

        HumanMessage(
            content=combined_input
        )
    ]


    result = llm.invoke(messages)


    # ChatGoogleGenerativeAI returns AIMessage
    answer = result.content


    # --------------------------------------------------
    # Step 5: Save conversation
    # --------------------------------------------------

    chat_history.append(
        HumanMessage(
            content=user_question
        )
    )

    chat_history.append(
        AIMessage(
            content=answer
        )
    )


    print(f"\nAnswer: {answer}")


    return answer


# --------------------------------------------------
# Simple chat loop
# --------------------------------------------------

def start_chat():

    print("Ask me questions! Type 'quit' to exit.")


    while True:

        question = input("\nYour question: ")


        if question.lower() == "quit":

            print("Goodbye!")

            break


        ask_question(question)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    start_chat()