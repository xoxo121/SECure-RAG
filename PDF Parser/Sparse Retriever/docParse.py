import logging
import time
import re
import pathway as pw
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from litellm import completion
from dotenv import load_dotenv
from typing import Dict
import aiohttp
from pathway.internals import udfs
from aiofiles import open as aio_open
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
import logging
import re
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from litellm import completion
import os
from replicate import Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# GROK_API_keys = os.getenv('GROQ_API_KEY').split(",")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN").split(",")
from pydantic import BaseModel


SYSTEM_PROMPT_TABLE = """You are tasked with summarizing the given table. Follow the instructions below:
            1. Provide the summary in sentences.
            2. Include important text and numbers in the summary.
            3. The summary should capture the gist of the entire table.
            4. Do not omit critical details from the table.

            The table is give below
        """


class Node(BaseModel):
    text: str


IMAGE_RESOLUTION_SCALE = 2.0

import time


class DoclingParser:
    def __init__(self, API_KEYS):
        self.api_keys = API_KEYS  # List of API keys
        self.current_api_key_index = 0  # To track the current API key

    def _get_current_api_key(self):
        return self.api_keys[self.current_api_key_index]

    def _set_next_api_key(self):
        self.current_api_key_index = (self.current_api_key_index + 1) % len(
            self.api_keys
        )
        # print(self.current_api_key_index)  # You can uncomment this for debugging

    def llm_call(self, model, model_input):
        retries = 3

        for attempt in range(retries):
            try:
                REPLICATE = Client(api_token=self._get_current_api_key())
                print(self._get_current_api_key())

                output = REPLICATE.run(model, input=model_input)

                return " ".join(output)

            except Exception as e:
                print(f"API call failed: {e}")
                self._set_next_api_key()  # Change to the next API key
                if attempt == retries - 1:
                    raise  # Reraise the error after the final attempt
                time.sleep(2)
        return " "

    def process_tables(self, table):
        model_input = {
            "top_p": 1,
            "prompt": SYSTEM_PROMPT_TABLE + "\n\n" + table,
            "temperature": 0,
            "max_new_tokens": 500,
        }
        # model = "google-deepmind/gemma-2b-it"
        model = "meta/meta-llama-3-8b-instruct"
        summary = self.llm_call(model=model, model_input=model_input)
        return summary.strip()

    @staticmethod
    def converter():
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = False
        pipeline_options.generate_table_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.do_ocr = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return doc_converter

    @staticmethod
    def remove_loc_tags(text):
        text = re.sub(r"<loc[^>]*>", "", text)
        text = re.sub(r"</loc[^>]*>", "", text)
        text = re.sub(r"<page_\d+>|</page_\d+>", "", text)
        return text

    @staticmethod
    def add_to_json_final(
        json_final, entry_type, header, cur_text=None, list_items=None
    ):
        if cur_text:
            json_final.append(
                {
                    "type": entry_type,
                    "title": header,
                    "text": f"{header}: {cur_text}",
                }
            )
        if list_items:
            list_items_s = " ".join(list_items)
            json_final.append(
                {
                    "type": "list",
                    "title": header,
                    "text": list_items_s,
                }
            )

    def table_column_aggregation(self, table_html):
        soup = BeautifulSoup(table_html, "html.parser")
        rows = soup.find_all("tr")
        largest_cells = []
        cells0 = rows[0]
        for cell in cells0:
            largest_cells.append(cell.get_text())
        for row in rows[1:]:
            cells = row.find_all("td")
            if cells:
                largest_cell = max(cells, key=lambda cell: len(cell.get_text()))
                largest_cells.append(largest_cell.get_text().strip())
        return " | ".join(largest_cells)

    def process_document(self, doc, html_content):
        table_iter = 0
        tables = doc.document.tables
        json_final = []
        soup = BeautifulSoup(html_content, "html.parser")

        header = ""
        cur_text = ""
        list_items = []
        table_summaries = []

        for tag in soup.document.children:
            if tag.name:
                if tag.name == "section_header":
                    self.add_to_json_final(json_final, "text", header, cur_text)
                    cur_text = ""
                    self.add_to_json_final(
                        json_final, "list", header, list_items=list_items
                    )
                    list_items = []
                    header = tag.text
                    if header == "Table of Contents":
                        header = ""

                elif tag.name == "text":
                    self.add_to_json_final(
                        json_final, "list", header, list_items=list_items
                    )
                    list_items = []
                    cur_text += tag.text

                elif tag.name == "list_item":
                    list_items.append(tag.text)

                elif tag.name == "table":
                    self.add_to_json_final(json_final, "text", header, cur_text)
                    cur_text = ""
                    self.add_to_json_final(
                        json_final, "list", header, list_items=list_items
                    )
                    list_items = []

                    table = tables[table_iter].export_to_html()
                    # Append an entry with the table's index and its markdown representation
                    # table_summaries.append({'index': len(json_final), 'table': table})
                    json_final.append(
                        {
                            "type": "table",
                            "table": table,
                            "table_iter": table_iter,
                            "title": header,
                            "text": self.table_column_aggregation(table),
                        }
                    )
                    table_iter += 1

                elif tag.name == "figure":
                    self.add_to_json_final(json_final, "text", header, cur_text)
                    cur_text = ""
                    self.add_to_json_final(
                        json_final, "list", header, list_items=list_items
                    )
                    list_items = []

                    # # figure_summary = await process_figures(tag)
                    # json_final.append({
                    #     'type': 'figure',
                    #     'summary': "",
                    #     'figure_iter': figure_iter,
                    #     'title': header
                    # })

        # Add remaining content
        self.add_to_json_final(json_final, "text", header, cur_text)
        self.add_to_json_final(json_final, "list", header, list_items)

        # Process tables concurrently
        table_tasks = [self.process_tables(entry["table"]) for entry in table_summaries]
        table_results = [task for task in table_tasks]

        # Update JSON with summaries
        for i, summary in enumerate(table_results):
            json_final[table_summaries[i]["index"]]["summary"] = summary

        return json_final

    @staticmethod
    def get_chunks(elements, chunk_size, overlap_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=["\n\n", "\n", ". "],
        )
        chunks = []
        for element in elements:
            if element["type"] == "text":
                temp_chunks = text_splitter.split_text(element["text"])
                for chunk in temp_chunks:
                    chunks.append(
                        {
                            "type": "text",
                            "title": element["title"],
                            "text": f"{element['title']}{chunk}",
                        }
                    )
            else:
                chunks.append(element)
        return chunks

    def parse(self, pdf_bytes, chunk_size=1024, overlap=128):
        pdf_stream = BytesIO(pdf_bytes)
        document = DocumentStream(name="example.pdf", stream=pdf_stream)
        converter = self.converter()
        conv_res = converter.convert(document)

        doc_format = conv_res.document.export_to_document_tokens()
        cleaned_doc = self.remove_loc_tags(doc_format)
        json_doc = self.process_document(conv_res, cleaned_doc)

        return self.get_chunks(json_doc, chunk_size=1024, overlap_size=128)



from pathway.xpacks.llm.parsers import ParseUtf8


class OurParse(pw.UDF):
    """
    Parse PDFs using `open-parse library <https://github.com/Filimoa/open-parse>`_.

    When used in the
    `VectorStoreServer <https://pathway.com/developers/api-docs/pathway-xpacks-llm/
    vectorstore#pathway.xpacks.llm.vector_store.VectorStoreServer>`_,
    splitter can be set to ``None`` as OpenParse already chunks the documents.

    Args:
        - table_args: dictionary containing the table parser arguments. Needs to have key ``parsing_algorithm``,
            with the value being one of ``"llm"``, ``"unitable"``, ``"pymupdf"``, ``"table-transformers"``.
            ``"llm"`` parameter can be specified to modify the vision LLM used for parsing.
            Will default to ``OpenAI`` ``gpt-4o``, with markdown table parsing prompt.
            Default config requires ``OPENAI_API_KEY`` environment variable to be set.
            For information on other parsing algorithms and supported arguments check
            `the OpenParse documentation <https://filimoa.github.io/open-parse/processing/parsing-tables/overview/>`_.
        - image_args: dictionary containing the image parser arguments.
            Needs to have the following keys ``parsing_algorithm``, ``llm``, ``prompt``.
            Currently, only supported ``parsing_algorithm`` is ``"llm"``.
            ``"llm"`` parameter can be specified to modify the vision LLM used for parsing.
            Will default to ``OpenAI`` ``gpt-4o``, with markdown image parsing prompt.
            Default config requires ``OPENAI_API_KEY`` environment variable to be set.
        - parse_images: whether to parse the images from the PDF. Detected images will be
            indexed by their description from the parsing algorithm.
            Note that images are parsed with separate OCR model, parsing may take a while.
        - processing_pipeline: ``openparse.processing.IngestionPipeline`` that will post process
            the extracted elements. Can be set to Pathway defined ``CustomIngestionPipeline``
            by setting to ``"pathway_pdf_default"``,
            ``SamePageIngestionPipeline`` by setting to ``"merge_same_page"``,
            or any of the pipelines under the ``openparse.processing``.
            Defaults to ``CustomIngestionPipeline``.
        - cache_strategy: Defines the caching mechanism. To enable caching,
            a valid :py:class:``~pathway.udfs.CacheStrategy`` should be provided.
            Defaults to None.

    Example:

    >>> import pathway as pw
    >>> from pathway.xpacks.llm import llms, parsers, prompts
    >>> chat = llms.OpenAIChat(model="gpt-4o")
    >>> table_args = {
    ...    "parsing_algorithm": "llm",
    ...    "llm": chat,
    ...    "prompt": prompts.DEFAULT_MD_TABLE_PARSE_PROMPT,
    ... }
    >>> image_args = {
    ...     "parsing_algorithm": "llm",
    ...     "llm": chat,
    ...     "prompt": prompts.DEFAULT_IMAGE_PARSE_PROMPT,
    ... }
    >>> parser = parsers.OpenParse(table_args=table_args, image_args=image_args)
    """

    def __init__(
        self,
        table_args: dict | None = None,
        image_args: dict | None = None,
        parse_images: bool = False,
        processing_pipeline: None | str | None = None,
        cache_strategy: udfs.CacheStrategy | None = None,
    ):
        # with optional_imports("xpack-llm-docs"):
        #     import openparse  # noqa:F401
        #     from pypdf import PdfReader  # noqa:F401

        #     from ._openparse_utils import (
        #         CustomDocumentParser,
        #         CustomIngestionPipeline,
        #         SamePageIngestionPipeline,
        #     )

        super().__init__(cache_strategy=cache_strategy)

        self.doc_parser = DoclingParser(REPLICATE_API_TOKEN)

    def __wrapped__(self, contents: bytes) -> list[tuple[str, dict]]:

        # reader = PdfReader(stream=BytesIO(contents))
        # doc = openparse.Pdf(file=reader)

        try:
            parsed_content = self.doc_parser.parse(contents)

            logger.info(
                f"Docling Parser completed parsing, total number of nodes: {len(parsed_content)}"
            )

            docs = []
            for node in parsed_content:
                print(node)
                node_t = Node(text=node["text"])
                node.pop("text")
                metadata = dict(node)
                docs.append((node_t.model_dump()["text"], metadata))

            return docs
        except Exception as e:
            node_t = Node(text="No-content")
            metadata = dict()
            return [(node_t.model_dump()["text"], metadata)]

    def __call__(self, contents: pw.ColumnExpression) -> pw.ColumnExpression:
        """
        Parse the given PDFs.

        Args:
            - contents (ColumnExpression[bytes]): A column with PDFs to be parsed, passed as bytes.

        Returns:
            A column with a list of pairs for each query. Each pair is a text chunk and
            metadata, which in case of `OpenParse` is an empty dictionary.
        """
        return super().__call__(contents)
