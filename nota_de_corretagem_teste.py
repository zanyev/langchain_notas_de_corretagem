from vertexai import init
from langchain_google_vertexai import VertexAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
import json
from langchain_core.output_parsers import JsonOutputParser


class PasswordProtectedPyPDFLoader(PyPDFLoader):
    """Custom loader to handle both password-protected and normal PDFs."""

    def __init__(self, file_path: str, password: str = None):
        super().__init__(file_path)
        self.password = password

    def load(self):
        """Load and split the documents, decrypting if necessary."""
        reader = PdfReader(self.file_path)

        if reader.is_encrypted:
            if self.password:
                try:
                    reader.decrypt(self.password)
                except Exception as e:
                    raise ValueError(f"Failed to decrypt PDF: {e}")
            else:
                raise ValueError("PDF is encrypted but no password was provided.")

        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            metadata = {"source": self.file_path, "page": i}
            docs.append(self._create_document(text, metadata))
        return docs


if __name__ == "__main__":

    load_dotenv()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )

    # Inicializa o Vertex AI com seu projeto e região
    init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )

    PASSWORD_PDF = os.getenv("PASSWORD_PDF")

    # CARREGA PDF EXEMPLO
    loader = PyPDFLoader(
        "./notas_de_corretagem/NotaNegociacao-6122830-17-04-2025-0.pdf", # caminho para um exemplo de nota de negociacao que servirá como base para o modelo
        password=PASSWORD_PDF,
    )
    example_pdf_pages = loader.load_and_split()
    example_conteudo_pdf = "\n".join(page.page_content for page in example_pdf_pages)

    gabarito_conteudo_pdf = [
        {
            "Data pregão": "17/04/2025",
            "Mercadoria": "DI1 F27",
            "Vlr de Operação/Ajuste": "120,99",
            "D/C Operação/Ajuste": "C",
            "Total líquido da nota": "84,25",
            "D/C líquido da nota": "C",
        },
        {
            "Data pregão": "17/04/2025",
            "Mercadoria": "CCM K25",
            "Vlr de Operação/Ajuste": "81,00",
            "D/C Operação/Ajuste": "D",
            "Total líquido da nota": "103,85",
            "D/C líquido da nota": "D",
        },
    ]

    # 3. Montar o prompt de exemplo
    examples = [
        {
            "input": example_conteudo_pdf,
            "output": json.dumps(gabarito_conteudo_pdf, indent=2, ensure_ascii=False)
            .replace("{", "{{")
            .replace("}", "}}"),
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Texto do PDF:\n{input}\n\nExtraia as informações no formato JSON:\n{output}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Texto do novo PDF:\n{input}\n\nExtraia as informações no mesmo formato JSON:",
        input_variables=["input"],
    )

    # inicializa o LLM
    llm = VertexAI(model_name="gemini-2.0-flash", temperature=0)

    # Cria o parser de saida json
    parser = JsonOutputParser()

    chain = few_shot_prompt | llm | parser

    path_novo_pdf = "./notas_de_corretagem/nota_950901.pdf" # sua nota de negociação que deseja extrair os dados
    # 5. Para usar: novo PDF
    loader_novo = PyPDFLoader(path_novo_pdf) # caso tenha senha no seu pdf forner para a funcao
    nota_de_corretagem_doc = loader_novo.load_and_split()
    conteudo_novo = "\n".join(page.page_content for page in nota_de_corretagem_doc)
    resposta = chain.invoke({"input": conteudo_novo})

    with open(
        f'./output/{path_novo_pdf.split("/")[-1].split(".")[0]}.json',
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(resposta, f, indent=2, ensure_ascii=False)
