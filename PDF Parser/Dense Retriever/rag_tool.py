from langchain_community.vectorstores.pathway import PathwayVectorClient

client = PathwayVectorClient(
    port=8666,
    host="127.0.0.1"
)
  
print(client.get_vectorstore_statistics())
       