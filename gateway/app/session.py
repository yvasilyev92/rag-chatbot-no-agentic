"""
Session management with DynamoDB storage.
"""
import uuid
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import logging

import boto3
from botocore.exceptions import ClientError

from .config import get_settings
from .models import Message

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions stored in DynamoDB."""
    
    def __init__(self):
        """Initialize the session manager with DynamoDB client."""
        settings = get_settings()
        self.table_name = settings.dynamodb_table
        self.ttl_hours = settings.session_ttl_hours
        self.max_history_tokens = settings.max_history_tokens
        self.chars_per_token = settings.chars_per_token
        
        # Initialize DynamoDB client
        self.dynamodb = boto3.resource('dynamodb', region_name=settings.aws_region)
        self.table = self.dynamodb.Table(self.table_name)
        
        logger.info(f"SessionManager initialized with table: {self.table_name}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    def _generate_message_id(self) -> str:
        """Generate a sortable message ID using timestamp."""
        # Aware UTC; the strftime format intentionally omits any tz suffix
        # so wire format matches what the pre-deprecation utcnow() produced.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_suffix}"
    
    def _calculate_expiry(self) -> int:
        """Calculate TTL expiry timestamp."""
        expiry_time = datetime.now(timezone.utc) + timedelta(hours=self.ttl_hours)
        return int(expiry_time.timestamp())
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return len(text) // self.chars_per_token
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new conversation session.
        
        Args:
            metadata: Optional metadata to associate with the session
            
        Returns:
            Dict containing session_id, created_at, expires_at
        """
        session_id = self._generate_session_id()
        created_at = datetime.now(timezone.utc)
        expires_at_ts = self._calculate_expiry()
        # tz-aware so the returned datetime is unambiguously UTC,
        # matching the rest of the timestamps we hand back to callers.
        expires_at = datetime.fromtimestamp(expires_at_ts, tz=timezone.utc)
        
        # Store session metadata as the first item
        # Use '0000_metadata' so it sorts BEFORE timestamp-based message IDs
        # (timestamps start with '2...' which is > '0' in lexicographic order)
        item = {
            'session_id': session_id,
            'message_id': '0000_metadata',
            'created_at': int(created_at.timestamp()),
            'expires_at': expires_at_ts,
            'type': 'session_metadata',
            'metadata': metadata or {}
        }
        
        try:
            self.table.put_item(Item=item)
            logger.info(f"Created session: {session_id}")
            
            return {
                'session_id': session_id,
                'created_at': created_at,
                'expires_at': expires_at
            }
        except ClientError as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a session.
        
        Args:
            session_id: The session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Dict containing the stored message details
        """
        message_id = self._generate_message_id()
        created_at = datetime.now(timezone.utc)
        expires_at = self._calculate_expiry()
        
        item = {
            'session_id': session_id,
            'message_id': message_id,
            'role': role,
            'content': content,
            'created_at': int(created_at.timestamp()),
            'expires_at': expires_at,
            'type': 'message',
            'tokens_estimate': self._estimate_tokens(content)
        }
        
        if metadata:
            item['metadata'] = metadata
        
        try:
            self.table.put_item(Item=item)
            logger.debug(f"Added message to session {session_id}: {role}")
            
            return {
                'message_id': message_id,
                'role': role,
                'content': content,
                'created_at': created_at
            }
        except ClientError as e:
            logger.error(f"Failed to add message: {e}")
            raise
    
    def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve all messages for a session.
        
        Args:
            session_id: The session identifier
            limit: Optional limit on number of messages to retrieve
            
        Returns:
            List of Message objects in chronological order
        """
        try:
            # Query messages for the session
            response = self.table.query(
                KeyConditionExpression='session_id = :sid AND message_id > :mid',
                ExpressionAttributeValues={
                    ':sid': session_id,
                    ':mid': '0000_metadata'  # Skip metadata item (sorts before timestamps)
                },
                ScanIndexForward=True  # Chronological order
            )
            
            items = response.get('Items', [])
            
            # Filter only message items
            messages = [
                Message(
                    role=item['role'],
                    content=item['content'],
                    created_at=datetime.fromtimestamp(int(item['created_at']))
                )
                for item in items
                if item.get('type') == 'message'
            ]
            
            if limit:
                messages = messages[-limit:]
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except ClientError as e:
            logger.error(f"Failed to get messages: {e}")
            raise
    
    def get_session_history_with_token_limit(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """
        Get session history, trimming old messages to stay within a token limit.

        When `max_tokens` is None, uses `self.max_history_tokens` (the
        existing behavior). When provided, uses `min(max_tokens,
        self.max_history_tokens)` so `MAX_HISTORY_TOKENS` is preserved as
        a hard upper bound while the per-turn budget can tighten it further.

        A `max_tokens` of 0 (or negative) yields an empty list.

        Args:
            session_id: The session identifier
            max_tokens: Optional per-turn budget override

        Returns:
            List of Message objects within the effective token budget
        """
        effective_limit = (
            min(max_tokens, self.max_history_tokens)
            if max_tokens is not None
            else self.max_history_tokens
        )

        if effective_limit <= 0:
            return []

        messages = self.get_session_messages(session_id)

        if not messages:
            return []

        # Calculate total tokens
        total_tokens = sum(self._estimate_tokens(m.content) for m in messages)

        # If within limit, return all messages
        if total_tokens <= effective_limit:
            return messages

        # Trim oldest messages until within limit
        trimmed_messages = list(messages)
        while total_tokens > effective_limit and len(trimmed_messages) > 1:
            removed = trimmed_messages.pop(0)
            total_tokens -= self._estimate_tokens(removed.content)
            logger.debug(f"Trimmed message to stay within token limit")

        # Edge case: a single remaining message could itself exceed the budget.
        # We still return it (the latest user/assistant turn is more valuable
        # than nothing); main.py logs and tolerates the overshoot.
        logger.info(
            f"Trimmed session {session_id} from {len(messages)} to "
            f"{len(trimmed_messages)} messages ({total_tokens} tokens, "
            f"limit={effective_limit})"
        )
        return trimmed_messages
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        try:
            response = self.table.get_item(
                Key={
                    'session_id': session_id,
                    'message_id': '0000_metadata'
                }
            )
            return 'Item' in response
        except ClientError as e:
            logger.error(f"Failed to check session existence: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata and info.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session info dict or None if not found
        """
        try:
            response = self.table.get_item(
                Key={
                    'session_id': session_id,
                    'message_id': '0000_metadata'
                }
            )
            
            if 'Item' not in response:
                return None
            
            item = response['Item']
            return {
                'session_id': session_id,
                'created_at': datetime.fromtimestamp(int(item['created_at'])),
                'expires_at': datetime.fromtimestamp(int(item['expires_at'])),
                'metadata': item.get('metadata', {})
            }
        except ClientError as e:
            logger.error(f"Failed to get session info: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its messages.
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if deletion was successful
        """
        try:
            # Query all items for this session
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={':sid': session_id},
                ProjectionExpression='session_id, message_id'
            )
            
            items = response.get('Items', [])
            
            if not items:
                logger.warning(f"Session not found for deletion: {session_id}")
                return False
            
            # Delete all items in batches
            with self.table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(
                        Key={
                            'session_id': item['session_id'],
                            'message_id': item['message_id']
                        }
                    )
            
            logger.info(f"Deleted session {session_id} with {len(items)} items")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete session: {e}")
            raise
    
    def refresh_session_ttl(self, session_id: str) -> bool:
        """
        Slide session expiry by updating TTL on the metadata item only.

        New messages already get expires_at = now + SESSION_TTL_HOURS on write.
        Bumping every historical message each turn was O(n) UpdateItems and
        is unnecessary: DynamoDB TTL is per-item, and old turns may drop
        ~24h after they were written even if the session is still active.
        """
        new_expiry = self._calculate_expiry()

        try:
            self.table.update_item(
                Key={
                    'session_id': session_id,
                    'message_id': '0000_metadata',
                },
                UpdateExpression='SET expires_at = :exp',
                ExpressionAttributeValues={':exp': new_expiry},
            )
            logger.debug(f"Refreshed session TTL for {session_id}")
            return True
        except ClientError as e:
            logger.error(f"Failed to refresh session TTL: {e}")
            return False


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
