import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from docParse import OurParse
import os


data = pw.io.fs.read(
    "./to-process/",
    format="binary",
    mode="streaming",
    with_metadata=True,
)

# data = pw.io.gdrive.read(
#     object_id = "",
#     service_user_credentials_file="credentials.json",
#     mode="streaming",
#     with_metadata=True,
# )

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embedder = OpenAIEmbeddings(model="text-embedding-3-large")
ourp = OurParse()

vector_store = VectorStoreServer.from_langchain_components(
    data,
    embedder=embedder,
    parser=ourp,
)

vector_store.run_server(host="0.0.0.0", port=8666)
