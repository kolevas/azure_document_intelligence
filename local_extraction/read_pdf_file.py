import pdfplumber
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
import os
import io
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_and_order_content(file_path: str) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Extracts text paragraphs and tables from a PDF document, ordering them by their
    vertical position on each page, and then by page order.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        Tuple[float, List[Dict[str, Any]]]: A tuple containing:
            - float: The timestamp when the extraction process started.
            - List[Dict[str, Any]]: A list of dictionaries, where each dictionary
              represents an extracted element (paragraph or table) with its content,
              type, bounding box, page number, and an order key for sorting.
              Returns an empty list if the file is not found or an error occurs.
    """
    start_time = time.time() # Capture the start time immediately

    ordered_elements = []

    if not os.path.exists(file_path):
        logger.error(f"File not found at '{file_path}'")
        return start_time, []

    try:
        with pdfplumber.open(file_path) as pdf_plumber:
            # Using PyMuPDF (fitz) in parallel for more robust text extraction
            with fitz.open(file_path) as pdf_fitz:
                total_pages = len(pdf_plumber.pages)
                logger.info(f"Starting hybrid extraction for '{os.path.basename(file_path)}' ({total_pages} pages).")

                for page_num, plumber_page in enumerate(pdf_plumber.pages):
                    logger.info(f"--- Processing Page {page_num + 1}/{total_pages} ---")
                    
                    elements_on_page = []
                    fitz_page = pdf_fitz[page_num]

                    # --- 1. Extract tables using pdfplumber ---
                    tables = plumber_page.find_tables()
                    table_bboxes = [(t.bbox, t) for t in tables] 
                    logger.info(f"  Detected {len(tables)} tables on Page {page_num + 1} (using pdfplumber).")

                    for bbox, table_obj in table_bboxes:
                        try:
                            table_data = table_obj.extract() 
                            markdown_table = ""
                            if table_data:
                                # Format table data into Markdown for better readability/downstream use
                                headers = table_data[0] if table_data and len(table_data) > 0 else []
                                rows = table_data[1:] if table_data and len(table_data) > 1 else []
                                
                                if headers:
                                    markdown_table += "| " + " | ".join(str(h).strip() if h is not None else "" for h in headers) + " |\n"
                                    markdown_table += "|---" * len(headers) + "|\n" # Markdown table header separator
                                
                                for row in rows:
                                    markdown_table += "| " + " | ".join(str(c).strip() if c is not None else "" for c in row) + " |\n"
                            
                            elements_on_page.append({
                                'type': 'table',
                                'content': markdown_table.strip(), # Clean up whitespace
                                'raw_table_data': table_data, # Include raw data for potential further processing
                                'bbox': bbox, # Bounding box coordinates (x0, y0, x1, y1)
                                'page_num': page_num + 1, # 1-based page number
                                'order_key': bbox[1] # Use y0 for initial vertical sorting on the page
                            })
                        except Exception as e:
                            logger.warning(f"  Error extracting table on Page {page_num + 1} with bbox {bbox}: {e}")


                    # --- 2. Extract and process text into paragraphs using PyMuPDF ---
                    all_text_elements_with_pos = []
                    try:
                        # Get words with their bounding boxes
                        fitz_words = fitz_page.get_text("words")
                        logger.info(f"  Extracted {len(fitz_words)} words on Page {page_num + 1} (using PyMuPDF).")
                        
                        for word_obj in fitz_words:
                            x0, y0, x1, y1, text_content = word_obj[0], word_obj[1], word_obj[2], word_obj[3], word_obj[4]
                            all_text_elements_with_pos.append({
                                'text': text_content,
                                'x0': x0, 'y0': y0, 
                                'x1': x1, 'y1': y1
                            })
                    except Exception as e:
                        logger.warning(f"  Error extracting words/text elements on Page {page_num + 1} with PyMuPDF: {e}")

                    current_paragraph_elements = [] 
                    
                    if all_text_elements_with_pos:
                        # Sort text elements by their vertical position, then horizontal
                        all_text_elements_with_pos.sort(key=lambda x: (x['y0'], x['x0']))

                        for text_obj in all_text_elements_with_pos:
                            is_part_of_table = False
                            # Check if the text element falls within any detected table's bounding box
                            for table_bbox, _ in table_bboxes:
                                if (text_obj['x0'] >= table_bbox[0] and text_obj['y0'] >= table_bbox[1] and
                                    text_obj['x1'] <= table_bbox[2] and text_obj['y1'] <= table_bbox[3]):
                                    is_part_of_table = True
                                    break 

                            if is_part_of_table:
                                continue # Skip text that is part of a table (already extracted)
                            
                            # Heuristic to detect new paragraphs: significant vertical gap between lines
                            # The '12' here is an arbitrary threshold, might need tuning based on document layout
                            if current_paragraph_elements and (text_obj['y0'] - current_paragraph_elements[-1]['y1'] > 12): 
                                # If a new paragraph is detected, finalize the previous one
                                min_x = min(item['x0'] for item in current_paragraph_elements)
                                min_y = min(item['y0'] for item in current_paragraph_elements)
                                max_x = max(item['x1'] for item in current_paragraph_elements)
                                max_y = max(item['y1'] for item in current_paragraph_elements)

                                elements_on_page.append({
                                    'type': 'paragraph',
                                    'content': " ".join([item['text'] for item in current_paragraph_elements]).strip(),
                                    'bbox': (min_x, min_y, max_x, max_y),
                                    'page_num': page_num + 1,
                                    'order_key': min_y # Use y0 of the first line for ordering
                                })
                                current_paragraph_elements = [] # Reset for the next paragraph
                            
                            current_paragraph_elements.append(text_obj) # Add text element to current paragraph

                        # Add the last collected paragraph on the page, if any
                        if current_paragraph_elements:
                             min_x = min(item['x0'] for item in current_paragraph_elements)
                             min_y = min(item['y0'] for item in current_paragraph_elements)
                             max_x = max(item['x1'] for item in current_paragraph_elements)
                             max_y = max(item['y1'] for item in current_paragraph_elements)

                             elements_on_page.append({
                                'type': 'paragraph',
                                'content': " ".join([item['text'] for item in current_paragraph_elements]).strip(),
                                'bbox': (min_x, min_y, max_x, max_y),
                                'page_num': page_num + 1,
                                'order_key': min_y
                            })
                    else: 
                        logger.info(f"  No non-table text elements found for paragraph processing on Page {page_num + 1}.")

                    # Sort elements collected on the current page by their vertical position
                    elements_on_page.sort(key=lambda x: x['order_key']) 
                    ordered_elements.extend(elements_on_page) # Add to the overall list
                    
    except FileNotFoundError:
        logger.error(f"The file '{file_path}' was not found.")
        return start_time, []
    except Exception as e:
        logger.exception(f"An unexpected error occurred during PDF processing for '{os.path.basename(file_path)}'.")
        return start_time, []

    return start_time, ordered_elements

if __name__ == "__main__":
    # This block is for testing read_pdf_file.py in isolation
    # Replace with a valid PDF path on your system for testing
    test_pdf_path = "example.pdf" # Make sure this file exists for testing

    if os.path.exists(test_pdf_path):
        extraction_start_time, extracted_content = extract_and_order_content(test_pdf_path)
        print(f"\n--- Extraction Summary ---")
        print(f"Extraction started at: {time.ctime(extraction_start_time)}")
        print(f"Total extracted elements (paragraphs & tables): {len(extracted_content)}")
        
        print("\nFirst 5 extracted elements:")
        for i, element in enumerate(extracted_content[:5]):
            print(f"  Element {i+1}: Type='{element['type']}', Page={element['page_num']}, "
                  f"Content (first 100 chars)='{element['content'][:100]}...'")
    else:
        print(f"Test PDF not found at: {test_pdf_path}. Please update the path or create the file.")