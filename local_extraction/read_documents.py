

import os
from pathlib import Path
import json 
from document_reader import DocumentReader

if __name__ == "__main__":
    reader = DocumentReader()

    base_data_dir = Path("/Users/snezhanakoleva/praksa/azure_document_intelligence/poc_data")
   
    output_file_path = Path("/Users/snezhanakoleva/praksa/azure_document_intelligence/local_extraction/ingested_data.json")

    print(f"--- Starting Data Ingestion from: {base_data_dir} ---")

    docs = reader.process_directory(directory_path=str(base_data_dir))
    
    print("\n--- Data Ingestion Complete ---")
    print(f"Total documents prepared for output: {len(docs) if docs else 0}") 
    if docs: 
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=4, ensure_ascii=False)
            print(f"Successfully wrote {len(docs)} documents to {output_file_path}")
        except Exception as e:
            print(f"Error writing to output file {output_file_path}: {e}")
    else:
        print("No documents were processed, so no output file was generated.")