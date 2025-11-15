from langchain_community.vectorstores.pathway import PathwayVectorClient
import json

def test_pathway_vector_client():
    client = PathwayVectorClient(
        url="https://humpback-engaging-kite.ngrok-free.app"
        # port=8000,
        # host="127.0.0.1"
    )
  
    print(client.get_vectorstore_statistics())
    # data = client.get_input_files()
    # print(len(data))
    # # with open('sample.txt', 'w') as file:
    # #     file.write(str(data))
    # print(client.get_vectorstore_statistics())
    # print(client.get_input_files())
    data = client.similarity_search(query="acciona",k=1000)
    print(data)

    # s = ""

    # for doc in data:
    #     s += doc.page_content

    # with open('data.txt', 'w') as f:
    #     f.write(s)


if __name__ == "__main__":
    test_pathway_vector_client()
