import time
import os 
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from local_extraction.document_reader import DocumentReader
if __name__ == "__main__":
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_API_KEY")
    
    dirPath = "/Users/snezhanakoleva/praksa/azure_document_intelligence/poc_data"
    
    start_time = time.time()
    document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
    
    files = DocumentReader.scan_directory(directory_path=dirPath)


    for filePath in files:
        print(f"Processing file: {filePath}")
    
        with open(filePath, "rb") as fd:
            formUrl = fd.read()
            
            poller = document_analysis_client.begin_analyze_document("prebuilt-read", formUrl)
            result = poller.result()
            
            chunks = DocumentReader.split_text_into_chunks_generalized(text=result.content, chunk_size=1000, overlap=200)
            print(f"Total chunks created: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}: {chunk}")
            print("----------------------------------------")

    end_time = time.time()
    actual_processing_time = end_time - start_time
    print(f"Processing time: {actual_processing_time:.2f} seconds")
