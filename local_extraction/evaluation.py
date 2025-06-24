
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ProcessingMetrics:
    """Metrics for individual file processing"""
    file_path: str
    file_type: str
    file_size_bytes: int
    processing_time_seconds: float
    success: bool
    error_message: Optional[str] = None
    
    # Content metrics
    total_characters: int = 0
    total_words: int = 0
    total_chunks: int = 0
    
    # Quality metrics
    empty_chunks: int = 0
    chunk_size_variance: float = 0.0
    average_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    
    # File-specific metrics
    pages_processed: Optional[int] = None  # For PDFs
    tables_extracted: Optional[int] = None  # For PDFs/Excel
    sheets_processed: Optional[List[str]] = None  # For Excel
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchProcessingMetrics:
    """Aggregated metrics for batch processing"""
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float
    
    # Aggregated content metrics
    total_documents_created: int
    total_characters_processed: int
    total_words_processed: int
    
    # File type breakdown
    file_type_counts: Dict[str, int]
    file_type_success_rates: Dict[str, float]
    
    # Performance metrics
    average_processing_time_per_file: float
    processing_rate_files_per_second: float
    processing_rate_mb_per_second: float
    
    # Quality metrics
    chunk_size_distribution: Dict[str, float]  # percentiles
    content_quality_score: float  # 0-1 score based on various factors
    
    # Error analysis
    error_types: Dict[str, int]
    problematic_files: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

