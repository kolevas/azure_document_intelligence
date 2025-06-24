import csv
import os
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed # For parallel processing of rows
import logging
from evaluation import ProcessingMetrics
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _process_csv_row_chunk_internal(
    row_chunk: List[Dict[str, Any]],
    base_file_metadata: Dict[str, Any],
    content_columns: Optional[List[str]],
    metadata_columns: Optional[List[str]],
    include_original_row_data: bool,
    start_row_index: int, # To correctly assign original_row_index
    fieldnames_for_content: List[str], # Pass validated fieldnames to the worker
    fieldnames_for_metadata: List[str] # Pass validated fieldnames to the worker
) -> List[Dict[str, Any]]:
    """
    Internal helper function to process a chunk of CSV rows.
    Designed to be run in a separate process.
    """
    chunk_documents: List[Dict[str, Any]] = []

    for local_row_index, row_dict in enumerate(row_chunk):
        original_row_index = start_row_index + local_row_index

        content_parts = []
        for col in fieldnames_for_content: # Use the validated list
            value = row_dict.get(col)
            if value is not None and str(value).strip() != '':
                content_parts.append(f"{col}: {value}")

        content = " ".join(content_parts).strip()

        if not content:
            continue # Skip empty content rows

        metadata_for_document = {
            **base_file_metadata,
            'original_row_index': original_row_index,
        }

        if include_original_row_data:
            metadata_for_document['original_row_data'] = row_dict

        for col in fieldnames_for_metadata: # Use the validated list
            if col in row_dict:
                metadata_for_document[col] = row_dict[col]
        
        if content_columns and not metadata_columns: # If only content_columns specified, remove them from metadata
             for col in content_columns:
                 metadata_for_document.pop(col, None)

        chunk_documents.append({
            'content': content,
            'metadata': metadata_for_document
        })

    return chunk_documents

def read_csv_file(
    file_path: str,
    content_columns: Optional[List[str]] = None,
    metadata_columns: Optional[List[str]] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8',
    skip_empty_lines: bool = True,
    include_original_row_data: bool = False,
    row_processing_chunk_size: int = 1000, 
    max_workers: int = os.cpu_count() or 4 
) -> Union[List[Dict[str, Any]], None]:
    start_time = time.time()
    all_documents: List[Dict[str, Any]] = []

    try:
        base_file_metadata = {
            'file_type': 'csv',
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
        }

        all_rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = [] 

        with open(file_path, mode='r', newline='', encoding=encoding) as csvfile:
            if skip_empty_lines:
                csvfile_iter = (line for line in csvfile if line.strip())
            else:
                csvfile_iter = csvfile

            logging.info(f"Main process: Reading CSV file '{file_path}' into memory sequentially...")
            reader = csv.DictReader(csvfile_iter, delimiter=delimiter)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                logging.warning(f"CSV file '{file_path}' declared as having a header, but no headers were found or file is empty.")
                return None

            # Validate columns here ONCE for all rows
            if content_columns:
                invalid_content_cols = [col for col in content_columns if col not in fieldnames]
                if invalid_content_cols:
                    logging.error(f"Error: Specified content_columns not found in header: {invalid_content_cols}")
                    return None
                final_content_fieldnames = content_columns
            else:
                final_content_fieldnames = fieldnames # If no specific content_columns, use all

            if metadata_columns:
                invalid_metadata_cols = [col for col in metadata_columns if col not in fieldnames]
                if invalid_metadata_cols:
                    logging.error(f"Error: Specified metadata_columns not found in header: {invalid_metadata_cols}")
                    return None
                final_metadata_fieldnames = metadata_columns
            else:
                final_metadata_fieldnames = fieldnames # If no specific metadata_columns, use all

            for row_dict in reader:
                all_rows.append(row_dict)

        logging.info(f"Main process: Finished reading {len(all_rows)} rows into memory. Starting parallel processing...")

        row_chunks = [
            all_rows[i:i + row_processing_chunk_size]
            for i in range(0, len(all_rows), row_processing_chunk_size)
        ]


        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            current_row_idx = 0
            for chunk in row_chunks:
                futures.append(executor.submit(
                    _process_csv_row_chunk_internal, 
                    chunk,
                    base_file_metadata,
                    content_columns, 
                    metadata_columns, 
                    include_original_row_data,
                    current_row_idx, 
                    final_content_fieldnames, 
                    final_metadata_fieldnames
                ))
                current_row_idx += len(chunk)

            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    if chunk_results:
                        all_documents.extend(chunk_results)
                except Exception as exc:
                    logging.error(f"A row processing chunk generated an exception: {exc}")

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        return None
    except csv.Error as e:
        logging.error(f"CSV parsing error in file '{file_path}': {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading or processing the CSV file: {e}")
        return None

    logging.info(f"Main process: Finished parallel processing for '{file_path}'. Total documents: {len(all_documents)}")

    
    return start_time, all_documents
