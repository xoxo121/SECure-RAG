from tqdm import tqdm
from typing import Callable
import logging
import json
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import litellm
from litellm import acompletion
import replicate
# BM25 imports #########################################################################
import os
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
# haystack
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack import Pipeline
from haystack.components.writers import DocumentWriter 
# Bm25 imports ############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
litellm.set_verbose = False

async def get_groq_reponse(
    query: str, input_params: dict = {"model": "groq/llama3-8b-8192", "max_tokens": 500}
) -> str:
    retry = 0
    while retry < 3:
        try:
            response = await acompletion(
                messages=[{"role": "user", "content": query}],
                **input_params,
            )
            break

        except Exception as e:
            response = None
            retry += 1


    return response.choices[0].message.content


"""***********************************Reranker code starts**************************************************************"""

class BGEReranker_replicate:
    def __init__(self, model_name="yxzwayne/bge-reranker-v2-m3", model_id : str = "7f7c6e9d18336e2cbf07d88e9362d881d2fe4d6a9854ec1260f115cabc106a8c"):
        """
        Initialize the BGE reranker.
        """
        self.replicate_client = replicate.Client(api_token = os.environ["REPLICATE_API_KEY"])
        self.model = self.replicate_client.models.get(model_name)
        self.version = self.model.versions.get(model_id)

        
    def rerank(self, query : str, documents : List[str], top_k : int | None = None):
        """
        Rerank the candidates based on the similarity with the query using the BGE model.
        """
        input_query_pair =  json.dumps([[query,doc] for doc in documents])
        
        if top_k == None: top_k = len(documents)
            
        output = self.replicate_client.run(f"{self.model.owner}/{self.model.name}:{self.version.id}",  # Format: owner/model_name:version_id
                                input = {"input_list":input_query_pair},)
        
        reranked_documents = list(sorted(zip(output, documents),reverse=True))[:top_k]

        return [rer_doc[1] for rer_doc in reranked_documents]
    
class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", batch_size : int = 6):
        """
        Initialize the BGE reranker.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.batch_size = batch_size

    def rerank(self, query, documents, top_k=None):
        """
        Rerank the candidates based on the similarity with the query using the BGE model.
        """
        candidates = documents

        inputs = self.tokenizer(
            [query] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        temp_dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        inputs_dataset = DataLoader(temp_dataset, batch_size = self.batch_size, shuffle=False)


        if top_k == None:
            top_k = len(candidates)

        with torch.no_grad():

            logits = []

            for batch_idx, batch_data in tqdm(enumerate(inputs_dataset)):
                
                outputs = self.model(input_ids = batch_data[0].to(device), attention_mask = batch_data[1].to(device))
                logits += outputs.logits.squeeze().tolist() if type(outputs.logits.squeeze().tolist()) != float else [outputs.logits.squeeze().tolist()]

                # emptying torch
                torch.cuda.empty_cache()
            

        return [
            candidate
            for _, candidate in sorted(
                zip(logits, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
        ][:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize Cross Encoder Reranker
        """
        self.cross_encoder = CrossEncoder(model_name=model_name, device = device)

    def rerank(self, query, documents, top_k=5):
        quer_doc_pairs = [(query, d) for d in documents]
        scores = self.cross_encoder.predict(quer_doc_pairs)

        sorted_candidates = [
            candidate
            for _, candidate in sorted(
                zip(scores, documents), key=lambda x: x[0], reverse=True
            )
        ]
        torch.cuda.empty_cache()
        return sorted_candidates[:top_k]
"""***********************************Reranker code ends**************************************************************"""

"""***********************************Hyde doc generator code starts**************************************************"""


class Hyde_document_creator:

    def __init__(
        self,
        api_key: str = None,
        model: str = "groq/llama3-70b-8192",
        template: Callable | None = None,
    ) -> None:

        # initializing all the varaibles
        self.model = model
        self.api_key = api_key

        # template should be funciton should out a query based in input
        if template != None:
            self.HyDE_template = template
        else:
            self.HyDE_template = (
                lambda query: f"""Given a question, generate a paragraph of text that answers the question.   
                NOTE : Dont change if any date or numbers are mentioned 
                Question: {query}    Paragraph:"""
            )

    async def get_hyde_output(
        self, query: str, n_paragraphs: int = 1, max_token: int = 500
    ) -> list[str]:
        # n_paragraphs = no of paragraphs

        HyDE_output = []
        template = self.HyDE_template(query)

        for i in range(n_paragraphs):

            chat_completion = await get_groq_reponse(
                query=template, input_params={"model": self.model, "max_tokens": 500}
            )

            # storing the output
            HyDE_output.append(chat_completion)

        return HyDE_output


# Multi - query , document generator
# Multi - query , document generator
class Hyde_Multi_document_creator(Hyde_document_creator):

    def __init__(
        self,
        api_key: str,
        model: str = "groq/llama3-70b-8192",
        template: Callable | None = None,
    ):

        super(Hyde_Multi_document_creator, self).__init__(api_key=api_key, model=model)

        # template should be funciton should out a query based in input
        if template != None:
            self.HyDE_multi_query_template = template
        else:
            seperator = "&"
            self.HyDE_multi_query_template = (
                lambda query, n_similar: f""" 
            Given a query : {query}, 
            Without changing the context in large amount , Output {n_similar} queries sperated by {seperator} symbol to avoid confusion 
            Just output the questions seperated by {seperator} without any additional text
            NOTE : Dont change if any date or numbers are mentioned """
            )

    async def get_hyde_multi_output(
        self, query: str, n_similar: int = 2, max_token: int = 500
    ) -> list[str]:
        # n_paragraphs = no of paragraphs

        HyDE_multi_output = []
        n_similar_queries = [query]

        for j in range(1):

            output_of_multi_query = await get_groq_reponse(
                query=self.HyDE_multi_query_template(query, n_similar=n_similar),
                input_params={"model": self.model, "max_tokens": 500},
            )

            n_similar_queries += output_of_multi_query.split("&")

            # print(len(n_similar_queries)) # -> passed test

        for query in n_similar_queries:

            # storing the output
            chat_completion = await self.get_hyde_output(
                query=query, n_paragraphs=1, max_token=max_token
            )
            HyDE_multi_output.append(chat_completion[0])

        return HyDE_multi_output


"""***********************************Hyde doc generator code ends*****************************************************************"""

"""***********************************Hyde document reteiver pipeline code starts**************************************************"""
## Full Multi Hyde pipeline without reranker
class full_multi_hyde_without_mean_reranker_pipeline:

    def __init__(self, HyDE_generator : object,  
                 retriever : object, 
                 reranker : object,
                 top_k : int = 20,
                 top_k_rerank : int = 10,
                 top_k_table : int = 5,
                 bm_25 : int = None):

        self.hyde_generator = HyDE_generator
        self.retriever = retriever
        self.top_k = top_k
        self.reranker = reranker
        self.top_k_rerank = top_k_rerank
        self.bm_25_retriever = bm_25
        self.top_k_table = top_k_table
    
    async def get_documents(self, queries : list[str], n_similar : int = 2):
        
        answer = []

        for query in tqdm(queries):

            # getting Hyde documents
            multi_doc = await self.hyde_generator.get_hyde_multi_output(query, n_similar = n_similar)
            print("Multi document creation has been finished ")

            if self.bm_25_retriever is not None: bm_25_doc = self.bm_25_retriever.retrieve_documents(query, k = 15)
            else: bm_25_doc = []
            print("BM25 search has been finished... , total BM25 docs retrieved : ", len(bm_25_doc) )
            # retrieve 
            all_retreived = []
            content_hash = set()

            # adding the bm25 docs
            for doc in bm_25_doc:
                meta_data = f"metadatab : \n file name : {doc['metadata']['file_path'].split('/')[-1]} \n table : \nNo tabular found for the passage "
                all_retreived.append(doc['page_content'] + '\n' + meta_data)
                content_hash.add(doc['page_content'])

            
            for doc in multi_doc[:]:

                retrieved_para_1 = await self.retriever.asimilarity_search(doc, self.top_k)

                ## Table change
                retrieved_tables = await self.retriever.asimilarity_search(doc, self.top_k_table, metadata_filter = "type == `table`" )
                for i in retrieved_tables : i.metadata['path'] += "\\table"
                retrieved_para_1 += retrieved_tables
                ## Table change

                retrieved_para = []

                for i in retrieved_para_1: 
                    if i.page_content not in content_hash:

                        # getting the met data
                        # meta_data = {"metadata" : {"file name" : i.metadata['name'] , "table" : None} }
                        meta_data = f"metadatad : \n file name : {i.metadata['path']} \n table : \n"
                        if 'table' in i.metadata : meta_data += f"{i.metadata['table']}"
                        else: meta_data += "No tabular found for the passage "

                        retrieved_para.append(i.page_content + '\n' + str(meta_data))
                        content_hash.add(i.page_content)   # hashing the data , so it wont retrieve the same data multiple times

                all_retreived.extend(retrieved_para)        

            print(f"All documents has been retrieved , reranking has been started")
            # reranking and response
            # print(len(all_retreived), len(content_hash), "test in Multi hyde without mean")
            reranked_docs = self.reranker.rerank(documents = all_retreived, query = query, top_k = self.top_k_rerank)

            print(F"Reranking has been finished")
            torch.cuda.empty_cache()

            answer.append(reranked_docs)
        
        return answer

class full_multi_hyde_without_mean_reranker_pipeline_bm25:

    def __init__(self, HyDE_generator : object,  
                 retriever : object,
                 bm_25 : object, 
                 reranker : object,
                 top_k : int = 20,
                 top_k_rerank : int = 10,
                 top_k_table : int = 5):

        self.hyde_generator = HyDE_generator
        self.retriever = retriever
        self.top_k = top_k
        self.reranker = reranker
        self.top_k_rerank = top_k_rerank
        self.bm_25_retriever = bm_25
        self.top_k_table = top_k_table
    
    async def get_documents(self, queries : list[str], n_similar : int = 2):
        
        answer = []

        for query in tqdm(queries):

            # getting Hyde documents
            multi_doc = await self.hyde_generator.get_hyde_multi_output(query, n_similar = n_similar)
            print("Multi document creation has been finished ")

            if self.bm_25_retriever is not None: bm_25_doc = await self.bm_25_retriever.asimilarity_search(query, k = 10)
            else: bm_25_doc = []
            print("BM25 search has been finished... , total BM25 docs retrieved : ", len(bm_25_doc) )

            # retrieve 
            all_retreived = []
            content_hash = set()

            # adding the bm25 docs
            for doc in bm_25_doc:
                tabular_data = "No tabular data found for the passage" if 'table' not in doc.metadata else doc.metadata['table']
                meta_data = f"metadatab : \n file name : {doc.metadata['path'].split('.')[0]} \n table : \n {tabular_data}"
                all_retreived.append(str(doc.page_content) + '\n' + str(meta_data))
                content_hash.add(doc.page_content)

            
            for doc in multi_doc[:]:

                retrieved_para_1 = await self.retriever.asimilarity_search(doc, self.top_k)
                if self.bm_25_retriever is not None: bm_25_doc = await self.bm_25_retriever.asimilarity_search(query, k = 5, metadata_filter = "type == `table`")
                else: bm_25_doc = []
                for i in bm_25_doc : i.metadata['path'] += "\\table"
                retrieved_para_1 += bm_25_doc

                ## Table change
                retrieved_tables = await self.retriever.asimilarity_search(doc, self.top_k_table, metadata_filter = "type == `table`" )
                for i in retrieved_tables : i.metadata['path'] += "\\table"
                retrieved_para_1 += retrieved_tables
                ## Table change

                retrieved_para = []

                for i in retrieved_para_1: 
                    if i.page_content not in content_hash:

                        # getting the met data
                        meta_data = f"metadatad : \n file name : {i.metadata['path']} \n table : \n"
                        if 'table' in i.metadata : meta_data += f"{i.metadata['table']}"
                        else: meta_data += "No tabular found for the passage "

                        retrieved_para.append(i.page_content + '\n' + str(meta_data))
                        content_hash.add(i.page_content)   # hashing the data , so it wont retrieve the same data multiple times

                all_retreived.extend(retrieved_para)        

            print(f"All documents has been retrieved , reranking has been started")
            # reranking and response
            reranked_docs = self.reranker.rerank(documents = all_retreived, query = query, top_k = self.top_k_rerank)

            print(F"Reranking has been finished")
            torch.cuda.empty_cache()

            answer.append(reranked_docs)
        
        return answer
"""***********************************Hyde document reteiver pipeline code ends**************************************************"""

"""***********************************BM25 Tools*********************************************************************************"""
class BM25_retriever:

    def __init__(self,):

        print("Only .pdf and .txt documents can be used here...")

        self.document_store = InMemoryDocumentStore()
        self.document_writer = DocumentWriter(self.document_store)
        self.retriever = InMemoryBM25Retriever(document_store = self.document_store)
        self.__create_preprocessing_pipeline()
    
    def __create_preprocessing_pipeline(self,):

        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
        # pdf to document
        pdf_to_document = PyPDFToDocument()
        text_to_document = TextFileToDocument()

        # joining the documents
        document_joiner = DocumentJoiner()

        # document splitter and embedder
        document_cleaner = DocumentCleaner()
        document_splitter = DocumentSplitter(split_by="word", split_length=256, split_overlap=25)

        ### Creating a pipeline
        self.preprocessing_pipeline = Pipeline()

        self.preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
        self.preprocessing_pipeline.add_component(instance=text_to_document, name="text_file_converter")
        self.preprocessing_pipeline.add_component(instance=pdf_to_document, name="pypdf_converter")


        self.preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
        self.preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
        self.preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
        self.preprocessing_pipeline.add_component(instance=self.document_writer, name="document_writer")
        # connecting the pipeline

        # connecting the documents
        self.preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        self.preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        self.preprocessing_pipeline.connect("text_file_converter", "document_joiner")
        self.preprocessing_pipeline.connect("pypdf_converter", "document_joiner")

        # joining and embedding
        self.preprocessing_pipeline.connect("document_joiner", "document_cleaner")
        self.preprocessing_pipeline.connect("document_cleaner", "document_splitter")
        self.preprocessing_pipeline.connect("document_splitter", "document_writer")
    
    def add_document(self, document_path : str) -> None:
        self.preprocessing_pipeline.run({"file_type_router": {"sources": [document_path] }})
        print("Document added successfully...")
    
    def add_documents_from_folder(self, folder_path : str) -> None:
        # for processing
        for doc_name in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, doc_name)
            self.preprocessing_pipeline.run({"file_type_router": {"sources": [file_path] }})
        print("Documents added successfully...")
    
    def save_vectorstore_to_disk(self, save_path : str) -> None:
        self.document_store.save_to_disk(save_path)
        print("Saved to disk...")
    
    def load_vectorstore_from_disk(self, load_path : str) -> None:
        self.document_store = self.document_store.load_from_disk(load_path)
        self.retriever = InMemoryBM25Retriever(document_store = self.document_store)
        self.document_writer = DocumentWriter(self.document_store)
        self.__create_preprocessing_pipeline()
        print("Document Store loaded")

    def retrieve_documents(self, query : str, k : int = 10) -> List[Dict[str,str|float]]:
        docs = self.retriever.run(query = query, top_k = k)
        return [{'page_content' : d.content , 'metadata' : d.meta , 'score' : d.score} for d in docs['documents']]
    
"""***********************************BM25 Tools***************************************************************************"""