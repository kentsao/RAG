from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

model = AutoModelForCausalLM.from_pretrained("acrastt/Marx-3B-V2", trust_remote_code=True, device_map='auto')

tokenizer = AutoTokenizer.from_pretrained("acrastt/Marx-3B-V2", trust_remote_code=True)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

#Load and Split PDF
loader = PyPDFLoader("/path/to/pdf file")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
texts = text_splitter.split_documents(data)

#Load vectorDB
vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)

#LLM chain
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    max_new_tokens=512
)

llm = HuggingFacePipeline(pipeline=pipe)

#QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())

print(qa_chain("Who am I")['result'])