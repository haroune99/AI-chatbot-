Building an AI powered chatbot (RAG architecture) to retrieve data on some custom documents. The custom documents include structured and accurate PDFs as well as exchanges from a forum. For the latter, I summarized first every discussion/topic of the forum before feeding it to the model.

It includes all the process of storing, chunking, retrieving, as well as functions to add history to the conversation and re-ranking of the results retrieved by the LLM.

I used then streamlit to create the app for the model
