"""
Document management for RAG knowledge base.

Handles document upload (in-memory processing), metadata tracking in DynamoDB,
and background async processing (text extraction, chunking, embedding, indexing).
"""
import asyncio
import logging
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import boto3
from botocore.exceptions import ClientError

from .config import get_settings
from .rag import (
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_pages,
    chunk_markdown,
    embed_texts,
    get_opensearch_rag,
    bust_search_cache,
)

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document uploads and metadata in DynamoDB."""

    DOCUMENTS_TABLE_SUFFIX = "-documents"

    def __init__(self):
        settings = get_settings()
        # Use a separate DynamoDB table for document metadata
        # e.g. "vllm-conversations" -> "vllm-documents"
        base_name = settings.dynamodb_table.replace("-conversations", "")
        self.table_name = f"{base_name}-documents"
        self.region = settings.aws_region
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # Initialize DynamoDB
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)

        logger.info(f"DocumentManager initialized with table: {self.table_name}")

    def create_document_record(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        file_size: int
    ) -> Dict[str, Any]:
        """
        Create a document metadata record in DynamoDB.

        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_type: File type ('pdf' or 'txt'; md and csv are stored as txt)
            file_size: File size in bytes

        Returns:
            Document metadata dict
        """
        now = datetime.now(timezone.utc)
        item = {
            'document_id': document_id,
            'filename': filename,
            'file_type': file_type,
            'file_size': file_size,
            'status': 'processing',
            'chunk_count': 0,
            'uploaded_at': now.isoformat(),
            'created_at': int(now.timestamp()),
            'error': None
        }

        try:
            self.table.put_item(Item=item)
            logger.info(f"Created document record: {document_id} ({filename})")
            return item
        except ClientError as e:
            logger.error(f"Failed to create document record: {e}")
            raise

    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        error: Optional[str] = None
    ):
        """
        Update a document's processing status.

        Args:
            document_id: Document identifier
            status: New status ('processing', 'ready', 'failed')
            chunk_count: Number of indexed chunks
            error: Error message if failed
        """
        try:
            update_expr = "SET #s = :status, chunk_count = :chunks"
            expr_values = {
                ':status': status,
                ':chunks': chunk_count
            }
            expr_names = {'#s': 'status'}

            if error:
                update_expr += ", #e = :error"
                expr_values[':error'] = error
                expr_names['#e'] = 'error'

            self.table.update_item(
                Key={'document_id': document_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_values,
                ExpressionAttributeNames=expr_names
            )
            logger.info(f"Updated document {document_id} status to '{status}'")
        except ClientError as e:
            logger.error(f"Failed to update document status: {e}")

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document metadata dict or None
        """
        try:
            response = self.table.get_item(Key={'document_id': document_id})
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Failed to get document: {e}")
            return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the knowledge base.

        Returns:
            List of document metadata dicts
        """
        try:
            response = self.table.scan(
                ProjectionExpression='document_id, filename, file_type, file_size, #s, chunk_count, uploaded_at, #e',
                ExpressionAttributeNames={'#s': 'status', '#e': 'error'}
            )
            items = response.get('Items', [])

            # Handle pagination for large tables
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    ProjectionExpression='document_id, filename, file_type, file_size, #s, chunk_count, uploaded_at, #e',
                    ExpressionAttributeNames={'#s': 'status', '#e': 'error'},
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))

            # Sort by upload time (newest first)
            items.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
            return items

        except ClientError as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document record and its chunks from OpenSearch.

        Args:
            document_id: Document identifier

        Returns:
            True if document existed and was deleted
        """
        # Check if document exists
        doc = self.get_document(document_id)
        if not doc:
            return False

        # Delete chunks from OpenSearch
        try:
            opensearch_rag = get_opensearch_rag()
            opensearch_rag.delete_document_chunks(document_id)
        except Exception as e:
            logger.error(f"Failed to delete OpenSearch chunks: {e}")

        # The corpus changed - drop any cached search results so we don't
        # return chunks for a document that no longer exists.
        try:
            bust_search_cache()
        except Exception as e:
            logger.warning(f"Failed to bust RAG search cache after delete: {e}")

        # Delete metadata from DynamoDB
        try:
            self.table.delete_item(Key={'document_id': document_id})
            logger.info(f"Deleted document: {document_id}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete document record: {e}")
            return False


# =============================================================================
# Background Document Processing
# =============================================================================

async def process_document_background(
    document_id: str,
    filename: str,
    file_type: str,
    file_bytes: bytes,
    doc_manager: DocumentManager
):
    """
    Process a document in the background: extract text, chunk, embed, index.

    This runs as an asyncio task so the upload endpoint can return immediately.

    Args:
        document_id: Unique document identifier
        filename: Original filename
        file_type: 'pdf' or 'txt' (md and csv are stored as txt)
        file_bytes: Raw file content
        doc_manager: DocumentManager instance
    """
    logger.info(f"Starting background processing for document {document_id} ({filename})")
    settings = get_settings()

    try:
        is_markdown = filename.lower().endswith(".md")

        if is_markdown:
            # Markdown-aware path: parse frontmatter + records + sections,
            # attach a self-describing preamble to each chunk.
            logger.info(f"[{document_id}] Chunking markdown (frontmatter + sections)...")
            chunks = await asyncio.to_thread(
                chunk_markdown,
                file_bytes,
                filename,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            logger.info(f"[{document_id}] Created {len(chunks)} markdown chunks")
        else:
            # Step 1: Extract text (PDF / plain text path)
            logger.info(f"[{document_id}] Extracting text from {file_type}...")
            if file_type == "pdf":
                pages = await asyncio.to_thread(extract_text_from_pdf, file_bytes)
            else:
                pages = await asyncio.to_thread(extract_text_from_txt, file_bytes)

            if not pages:
                doc_manager.update_document_status(
                    document_id, "failed", error="No text could be extracted from the file"
                )
                return

            total_text_len = sum(len(p["text"]) for p in pages)
            logger.info(
                f"[{document_id}] Extracted {len(pages)} pages, {total_text_len} chars"
            )

            # Step 2: Chunk text
            logger.info(f"[{document_id}] Chunking text...")
            chunks = await asyncio.to_thread(
                chunk_pages, pages, settings.chunk_size, settings.chunk_overlap
            )

            # Attach a lightweight filename-derived preamble so non-markdown
            # chunks still get a (basic) self-describing prefix and an
            # indexable document_name field.
            doc_name = filename.rsplit(".", 1)[0] if "." in filename else filename
            preamble = f"[Document: {doc_name}]\n\n"
            for chunk in chunks:
                chunk["raw_text"] = chunk["text"]
                chunk["text"] = preamble + chunk["text"]
                chunk["document_name"] = doc_name

            logger.info(f"[{document_id}] Created {len(chunks)} chunks")

        if not chunks:
            doc_manager.update_document_status(
                document_id, "failed", error="Text extraction produced no usable chunks"
            )
            return

        # Step 3: Generate embeddings
        logger.info(f"[{document_id}] Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = await asyncio.to_thread(embed_texts, chunk_texts)
        logger.info(f"[{document_id}] Generated {len(embeddings)} embeddings")

        # Step 4: Index in OpenSearch
        logger.info(f"[{document_id}] Indexing chunks in OpenSearch...")
        opensearch_rag = get_opensearch_rag()
        indexed_count = await asyncio.to_thread(
            opensearch_rag.index_chunks, document_id, filename, chunks, embeddings
        )

        # The corpus just grew - cached search results from before this
        # document existed are now stale. Drop the search cache so the
        # next query reflects the new content.
        try:
            bust_search_cache()
        except Exception as e:
            logger.warning(f"Failed to bust RAG search cache after index: {e}")

        # Step 5: Update status
        doc_manager.update_document_status(document_id, "ready", chunk_count=indexed_count)
        logger.info(
            f"[{document_id}] Processing complete: {indexed_count} chunks indexed"
        )

    except Exception as e:
        logger.error(f"[{document_id}] Processing failed: {e}", exc_info=True)
        doc_manager.update_document_status(
            document_id, "failed", error=str(e)
        )


# =============================================================================
# Global Document Manager Instance
# =============================================================================

_document_manager: Optional[DocumentManager] = None


def get_document_manager() -> DocumentManager:
    """Get or create the global document manager instance."""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
