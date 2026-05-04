"""Column-name constants for raw and processed datasets.

Using named constants helps reduce typo-related bugs across scripts.
"""

from __future__ import annotations

# Raw events columns from Retailrocket.
RAW_TIMESTAMP_COL = "timestamp"
RAW_VISITOR_COL = "visitorid"
RAW_EVENT_COL = "event"
RAW_ITEM_COL = "itemid"
RAW_TRANSACTION_COL = "transactionid"

# Processed columns used throughout the project.
SESSION_ID_COL = "session_id"
VISITOR_ID_COL = "visitor_id"
ITEM_ID_COL = "item_id"
ITEM_IDX_COL = "item_idx"
TIMESTAMP_COL = "timestamp"
EVENT_TYPE_COL = "event_type"
POSITION_COL = "position"
SESSION_LENGTH_COL = "session_length"
SESSION_START_COL = "session_start_ts"
SESSION_END_COL = "session_end_ts"
