import os
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import statistics
import re
from bs4 import BeautifulSoup
import markdown
import docx
import pandas as pd
import logging
import concurrent.futures # Import for threading

# Import the PDF extraction function from the separate file
from .read_pdf_file import extract_and_order_content

# Dummy implementations for demonstration if full modules are not provided
class ProcessingMetrics:
    def __init__(self, file_path, file_type, file_size_bytes, processing_time_seconds, success, **kwargs):
        self.file_path = file_path
        self.file_type = file_type
        self.file_size_bytes = file_size_bytes
        self.processing_time_seconds = processing_time_seconds
        self.success = success
        self.error_message = kwargs.get('error_message')
        self.total_characters = kwargs.get('total_characters', 0)
        self.total_words = kwargs.get('total_words', 0)
        self.total_chunks = kwargs.get('total_chunks', 0)
        self.empty_chunks = kwargs.get('empty_chunks', 0)
        self.chunk_size_variance = kwargs.get('chunk_size_variance', 0.0)
        self.average_chunk_size = kwargs.get('average_chunk_size', 0.0)
        self.min_chunk_size = kwargs.get('min_chunk_size', 0)
        self.max_chunk_size = kwargs.get('max_chunk_size', 0)

    def to_dict(self):
        return {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size_bytes': self.file_size_bytes,
            'processing_time_seconds': self.processing_time_seconds,
            'success': self.success,
            'error_message': self.error_message,
            'total_characters': self.total_characters,
            'total_words': self.total_words,
            'total_chunks': self.total_chunks,
            'empty_chunks': self.empty_chunks,
            'chunk_size_variance': self.chunk_size_variance,
            'average_chunk_size': self.average_chunk_size,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size
        }

class BatchProcessingMetrics:
    # This class would typically aggregate metrics across all files processed in a batch
    # For this example, it's still a placeholder.
    def to_dict(self):
        return {"message": "Batch metrics not fully implemented in this example."}


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:
    logger.warning("Warning: 'tiktoken' not found. Token-based chunking will fall back to character-based. Install with 'pip install tiktoken'.")
    tiktoken = None


class DocumentReader:
    def __init__(self):  
        self.supported_extensions = {'.txt', '.pdf', '.md', '.html', '.docx', '.xlsx', '.pptx', '.csv'}
        self.processing_history: List[ProcessingMetrics] = []
        self.batch_metrics: Optional[BatchProcessingMetrics] = None # Still a dummy for now

    def get_processing_history(self) -> List[Dict[str, Any]]:
        return [metrics.to_dict() for metrics in self.processing_history]
    
    def get_batch_metrics(self) -> Optional[Dict[str, Any]]:
        return self.batch_metrics.to_dict() if self.batch_metrics else None
    
    def clear_metrics(self):
        self.processing_history.clear()
        self.batch_metrics = None
    
    def _calculate_content_metrics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not documents:
            return {
                'total_characters': 0, 'total_words': 0, 'total_chunks': 0,
                'empty_chunks': 0, 'chunk_size_variance': 0.0,
                'average_chunk_size': 0.0, 'min_chunk_size': 0, 'max_chunk_size': 0
            }
        
        chunk_sizes = [len(doc.get('content', '')) for doc in documents]
        total_chars = sum(chunk_sizes)
        total_words = sum(len(doc.get('content', '').split()) for doc in documents)
        empty_chunks = sum(1 for doc in documents if not doc.get('content', '').strip())
        
        avg_chunk_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
        chunk_variance = statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0
        
        return {
            'total_characters': total_chars, 'total_words': total_words, 'total_chunks': len(documents),
            'empty_chunks': empty_chunks, 'chunk_size_variance': chunk_variance,
            'average_chunk_size': avg_chunk_size, 'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        score = 1.0
        if metrics['total_chunks'] > 0:
            empty_ratio = metrics['empty_chunks'] / metrics['total_chunks']
            score -= empty_ratio * 0.3
        if metrics['average_chunk_size'] > 0:
            cv = (metrics['chunk_size_variance'] ** 0.5) / metrics['average_chunk_size']
            score -= min(cv * 0.2, 0.3)
        avg_size = metrics['average_chunk_size']
        if avg_size > 0:
            if avg_size < 100: score -= 0.2
            elif avg_size > 5000: score -= 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def split_text_into_chunks_generalized(
        text: str,
        chunk_size: int = 1500,
        overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function=len
    ) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        final_chunks = []
        current_splits = [text]

        for separator in separators:
            if not current_splits:
                break
            
            temp_splits = []
            for segment in current_splits:
                if length_function(segment) > chunk_size:
                    if separator:
                        new_sub_segments = segment.split(separator)
                    else:
                        new_sub_segments = list(segment)

                    buffer = []
                    for sub_segment in new_sub_segments:
                        sub_segment = sub_segment.strip()
                        if not sub_segment:
                            continue
                        
                        candidate = (buffer[-1] + (separator if separator else '') + sub_segment) if buffer else sub_segment
                        if buffer and length_function(candidate) <= chunk_size:
                             buffer[-1] = candidate
                        else:
                            if buffer:
                                temp_splits.extend(buffer)
                            buffer = [sub_segment]
                    if buffer:
                        temp_splits.extend(buffer)
                else:
                    temp_splits.append(segment)
            current_splits = temp_splits
            if all(length_function(s) <= chunk_size for s in current_splits):
                break

        text_to_chunk = "\n".join(s for s in current_splits if s.strip()) 
        
        start = 0
        while start < length_function(text_to_chunk):
            end = start + chunk_size
            
            if end < length_function(text_to_chunk):
                temp_chunk_candidate = text_to_chunk[start:end]
                break_points = []
                for sep in [".", "\n", " "]:
                    last_pos = temp_chunk_candidate.rfind(sep)
                    if last_pos != -1:
                        break_points.append(last_pos)
                
                split_point_in_candidate = -1
                if break_points:
                    split_point_in_candidate = max(break_points)
                
                if split_point_in_candidate != -1:
                    end = start + split_point_in_candidate + 1
                
            else:
                end = length_function(text_to_chunk)

            chunk = text_to_chunk[start:end]
            final_chunks.append(chunk.strip()) 
            
            if end == length_function(text_to_chunk):
                break
            
            start = end - overlap
            start = max(0, start)
            
        # print(f"Text splitting completed. Total final chunks: {len(final_chunks)}") # Suppressed for cleaner multi-threading output
        return final_chunks

    def _get_length_function(self, token_based_chunking: bool):
        if token_based_chunking and tiktoken:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
                # logger.info(f"Using token-based chunking with '{encoder.name}' tokenizer.") # Suppressed for cleaner multi-threading output
                return lambda t: len(encoder.encode(t))
            except Exception as e:
                logger.warning(f"Could not load tiktoken encoder for token-based chunking ({e}). Falling back to character-based.")
        # logger.info("Using character-based chunking.") # Suppressed for cleaner multi-threading output
        return len

    def _read_raw_content(self, file_path: str, file_extension: str) -> Tuple[Union[str, List[Dict[str, Any]], Dict[str, Any], None], Dict[str, Any]]:
        """Helper to read raw content and generate base metadata."""
        base_file_metadata = {
            'file_name': os.path.basename(file_path),
            'file_type': file_extension.lstrip('.')
        }
        raw_content: Any = None
        
        try:
            file_size = os.path.getsize(file_path)
            base_file_metadata['file_size'] = file_size

            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                # Clean Project Gutenberg headers/footers
                start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
                end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
                start_index = full_text.find(start_marker)
                if start_index != -1:
                    start_index = full_text.find('\n', start_index) + 1
                else:
                    start_index = 0
                end_index = full_text.rfind(end_marker)
                if end_index == -1:
                    end_index = len(full_text)
                raw_content = full_text[start_index:end_index]

            elif file_extension == '.pdf':
                _start_time, ordered_elements = extract_and_order_content(file_path=file_path)
                raw_content = ordered_elements

            elif file_extension == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, "html.parser")
                raw_content = soup.get_text(separator=' ', strip=True)

            elif file_extension == '.html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, "html.parser")
                title_tag = soup.find('title')
                base_file_metadata['title'] = title_tag.get_text(strip=True) if title_tag else None
                raw_content = soup.get_text(separator=' ', strip=True)

            elif file_extension == '.docx':
                doc = docx.Document(file_path)
                raw_content = '\n'.join([para.text for para in doc.paragraphs])

            elif file_extension == '.xlsx':
                xls = pd.ExcelFile(file_path)
                all_sheets_data = {}
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    all_sheets_data[sheet_name] = df.to_string(index=False)
                raw_content = all_sheets_data # Store as dict to preserve sheet names
                base_file_metadata['sheet_names'] = xls.sheet_names

            elif file_extension == '.csv':
                # Dummy implementation for CSV - needs actual reading logic
                raw_content = "This is dummy CSV content.\nRow1,Col1,Col2\nRow2,ValA,ValB"
                logger.warning("CSV reading is a dummy implementation. Replace with actual CSV parsing.")
            
            elif file_extension == '.pptx':
                # Dummy implementation for PPTX - needs 'python-pptx'
                raw_content = "This is dummy content from a PPTX file. Actual extraction requires 'python-pptx'."
                logger.warning("PPTX reading is a dummy implementation. Install 'python-pptx' for actual parsing.")

            else:
                logger.warning(f"Unsupported file type: {file_extension} for file: {file_path}")
                return None, base_file_metadata

            return raw_content, base_file_metadata

        except FileNotFoundError:
            logger.error(f"File not found at {file_path}")
            base_file_metadata['file_size'] = 0 # Update if not found
            return None, base_file_metadata
        except Exception as e:
            logger.exception(f"Error reading {file_extension} file {file_path}")
            base_file_metadata['file_size'] = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            return None, base_file_metadata

    def _process_and_chunk(self, raw_content: Any, base_metadata: Dict[str, Any], chunk_size: int, overlap: int, token_based_chunking: bool) -> List[Dict[str, Any]]:
        """Helper to apply chunking logic and return documents."""
        documents: List[Dict[str, Any]] = []
        file_type = base_metadata['file_type']
        length_func = self._get_length_function(token_based_chunking)
        
        if raw_content is None:
            return []

        if file_type == 'pdf' and isinstance(raw_content, list): # Specific semantic chunking for PDF
            current_chunk_content_parts = []
            current_chunk_length = 0
            chunk_index = 0

            for element in raw_content: # raw_content here is ordered_elements
                element_content = element.get('content', '').strip()
                if not element_content:
                    continue

                element_length = length_func(element_content)
                effective_element_length = element_length + (length_func("\n\n") if current_chunk_content_parts else 0)

                if current_chunk_content_parts and (current_chunk_length + effective_element_length > chunk_size):
                    combined_chunk_text = "\n\n".join(current_chunk_content_parts).strip()
                    if combined_chunk_text:
                        documents.append({
                            'content': combined_chunk_text,
                            'metadata': {
                                **base_metadata,
                                'chunk_type': f"{file_type}_chunk",
                                'chunk_index': chunk_index,
                                'chunk_length': length_func(combined_chunk_text),
                                'num_words': len(combined_chunk_text.split())
                            }
                        })
                        chunk_index += 1
                    current_chunk_content_parts = []
                    current_chunk_length = 0

                if element_length > chunk_size:
                    if current_chunk_content_parts:
                        combined_chunk_text = "\n\n".join(current_chunk_content_parts).strip()
                        if combined_chunk_text:
                            documents.append({
                                'content': combined_chunk_text,
                                'metadata': {
                                    **base_metadata,
                                    'chunk_type': f"{file_type}_chunk",
                                    'chunk_index': chunk_index,
                                    'chunk_length': length_func(combined_chunk_text),
                                    'num_words': len(combined_chunk_text.split())
                                }
                            })
                            chunk_index += 1
                        current_chunk_content_parts = []
                        current_chunk_length = 0

                    sub_chunks = DocumentReader.split_text_into_chunks_generalized(
                        text=element_content,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        length_function=length_func
                    )
                    for sub_chunk_content in sub_chunks:
                        if sub_chunk_content.strip():
                            documents.append({
                                'content': sub_chunk_content.strip(),
                                'metadata': {
                                    **base_metadata,
                                    'chunk_type': f"{file_type}_sub_chunk_from_{element['type']}",
                                    'chunk_index': chunk_index,
                                    'chunk_length': length_func(sub_chunk_content.strip()),
                                    'num_words': len(sub_chunk_content.strip().split()),
                                    'original_page_num_of_element': element['page_num'],
                                    'original_element_type': element['type']
                                }
                            })
                            chunk_index += 1
                    current_chunk_content_parts = []
                    current_chunk_length = 0
                else:
                    current_chunk_content_parts.append(element_content)
                    current_chunk_length += effective_element_length
            
            if current_chunk_content_parts:
                combined_chunk_text = "\n\n".join(current_chunk_content_parts).strip()
                if combined_chunk_text:
                    documents.append({
                        'content': combined_chunk_text,
                        'metadata': {
                            **base_metadata,
                            'chunk_type': f"{file_type}_chunk",
                            'chunk_index': chunk_index,
                            'chunk_length': length_func(combined_chunk_text),
                            'num_words': len(combined_chunk_text.split())
                        }
                    })

        elif file_type == 'xlsx' and isinstance(raw_content, dict): # Handle XLSX specific raw_content (dict of sheets)
            full_text_for_chunking = ""
            for sheet_name, sheet_data in raw_content.items():
                full_text_for_chunking += f"\n--- Sheet: {sheet_name} ---\n"
                full_text_for_chunking += sheet_data + "\n"
            
            text_chunks = DocumentReader.split_text_into_chunks_generalized(
                text=re.sub(r'\s+', ' ', full_text_for_chunking).strip(),
                chunk_size=chunk_size,
                overlap=overlap,
                length_function=length_func
            )
            for i, chunk_content in enumerate(text_chunks):
                processed_chunk_content = chunk_content.lower().strip()
                if processed_chunk_content:
                    documents.append({
                        'content': processed_chunk_content,
                        'metadata': {
                            **base_metadata,
                            'chunk_type': f"{file_type}_chunk",
                            'chunk_index': i,
                            'chunk_length': length_func(processed_chunk_content),
                            'num_words': len(processed_chunk_content.split())
                        }
                    })

        elif isinstance(raw_content, str): # Generic text-based chunking for TXT, MD, HTML, DOCX, CSV, PPTX
            text_chunks = DocumentReader.split_text_into_chunks_generalized(
                text=re.sub(r'\s+', ' ', raw_content).strip(),
                chunk_size=chunk_size,
                overlap=overlap,
                length_function=length_func
            )
            for i, chunk_content in enumerate(text_chunks):
                processed_chunk_content = chunk_content.lower().strip()
                if processed_chunk_content:
                    documents.append({
                        'content': processed_chunk_content,
                        'metadata': {
                            **base_metadata,
                            'chunk_type': f"{file_type}_chunk",
                            'chunk_index': i,
                            'chunk_length': length_func(processed_chunk_content),
                            'num_words': len(processed_chunk_content.split())
                        }
                    })
        else:
            logger.error(f"Unsupported raw content type for chunking: {type(raw_content)} for file type {file_type}")

        return documents

    def read_single_document(
        self,
        file_path: str,
        chunk_size: int = 1500,
        overlap: int = 200,
        token_based_chunking: bool = False
       
    ) -> Tuple[Union[List[Dict[str, Any]], None], ProcessingMetrics]:
        """
        Reads a single document, processes it, and returns a tuple of (documents, metrics).
        Returns (None, metrics_with_error) if processing fails.
        """
        start_time = time.time()
        file_extension = Path(file_path).suffix.lower()

        raw_content, base_metadata = self._read_raw_content(file_path, file_extension)
        
        success = False
        error_message = None
        documents: List[Dict[str, Any]] = []

        if raw_content is not None and (isinstance(raw_content, str) and raw_content.strip() or not isinstance(raw_content, str)):
            documents = self._process_and_chunk(raw_content, base_metadata, chunk_size, overlap, token_based_chunking)
            success = True
            if not documents:
                error_message = "No chunks generated after processing."
                success = False
        else:
            if raw_content is None: # File not found or reading error
                error_message = base_metadata.get('error_message', "File could not be read or is empty.")
            else: # Content was empty after initial extraction/cleaning (for string types)
                error_message = "No meaningful content extracted from the file."
            
            success = False

        processing_time = time.time() - start_time
        content_metrics = self._calculate_content_metrics(documents)
        
        metrics = ProcessingMetrics(
            file_path=file_path,
            file_type=base_metadata['file_type'],
            file_size_bytes=base_metadata.get('file_size', 0),
            processing_time_seconds=processing_time,
            success=success,
            error_message=error_message,
            **content_metrics
        )
        
        if not success:
            logger.error(f"Failed to process '{file_path}': {error_message}")
            return None, metrics
        
        logger.info(f"'{file_path}' processed into {len(documents)} chunks (took {processing_time:.2f}s).")
        return documents, metrics
    
    @staticmethod
    def scan_directory( directory_path: str, recursive: bool = True) -> List[str]:
        document_files = []
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory not found or is not a directory: {directory_path}")
            return []

        if recursive:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {'.txt', '.pdf', '.md', '.html', '.docx', '.xlsx', '.pptx', '.csv'}:
                    document_files.append(str(file_path))
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in {'.txt', '.pdf', '.md', '.html', '.docx', '.xlsx', '.pptx', '.csv'}:
                    document_files.append(str(file_path))
        return document_files

    def process_directory(self, directory_path: str,
                         recursive: bool = True, chunk_size: int = 1500, overlap: int = 200,
                         token_based_chunking: bool = False,
                         max_workers: int = os.cpu_count() or 1 # Number of threads
                         ) -> List[Dict[str, Any]]: 
        print(f"Scanning directory: {directory_path}")
        document_files = self.scan_directory(directory_path, recursive)
        print(f"Found {len(document_files)} supported files")

        documents_to_add = []
        self.processing_history.clear() # Clear history before batch processing

        batch_start_time = time.time() # Start timing for the entire batch

        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for each call to read_single_document
            future_to_file = {
                executor.submit(
                    self.read_single_document, 
                    file_path, 
                    chunk_size, 
                    overlap, 
                    token_based_chunking
                ): file_path 
                for file_path in document_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks, metrics = future.result() # Get results from the completed thread
                    self.processing_history.append(metrics) # Add metrics to history
                    if chunks:
                        documents_to_add.extend(chunks)
                    # Individual completion messages are now handled by read_single_document's logger.info
                except Exception as exc:
                    logger.error(f"'{file_path}' generated an exception: {exc}")
                    # Create a failed metric entry for consistency
                    failed_metrics = ProcessingMetrics(
                        file_path=file_path,
                        file_type=Path(file_path).suffix.lower().lstrip('.'),
                        file_size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        processing_time_seconds=time.time() - batch_start_time, # Rough time until error detected
                        success=False,
                        error_message=f"Exception during processing: {exc}"
                    )
                    self.processing_history.append(failed_metrics)

        batch_end_time = time.time()
        total_batch_time_seconds = batch_end_time - batch_start_time

        print(f"\n--- Batch Processing Summary ---")
        print(f"Total processed chunks from {len(self.processing_history)} files: {len(documents_to_add)}") 
        print(f"Total wall-clock time for batch processing: {total_batch_time_seconds:.2f} seconds")
        print ("\nDetailed Processing metrics per file:")
        for metrics in self.processing_history:
            print(f"  File: {metrics.file_path}, Type: {metrics.file_type}, Size: {metrics.file_size_bytes} bytes, "
                  f"Time: {metrics.processing_time_seconds:.2f}s, Success: {metrics.success}, "
                  f"Chunks: {metrics.total_chunks}, Empty Chunks: {metrics.empty_chunks}, "
                  f"Avg Chunk Size: {metrics.average_chunk_size:.2f}, Quality Score: {self._calculate_quality_score(metrics.to_dict()):.2f}")
        
        if self.batch_metrics:
            print(f"Batch Metrics: {self.batch_metrics.to_dict()}") 
        else:
            print("Note: Batch metrics aggregation is not fully implemented in this example (only per-file metrics are).")

        return documents_to_add
