"""Initial sentiment_records table.

Revision ID: 001_initial_sentiment_records
Revises:
Create Date: 2026-11-20 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_sentiment_records"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sentiment_records",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("topic", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("sentiment_label", sa.String(length=32), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("sentiment_score", sa.Float(), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sentiment_records_topic",
        "sentiment_records",
        ["topic"],
        unique=False,
    )
    op.create_index(
        "ix_sentiment_records_sentiment_label",
        "sentiment_records",
        ["sentiment_label"],
        unique=False,
    )
    op.create_index(
        "ix_sentiment_records_created_at",
        "sentiment_records",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_sentiment_records_topic_created_at",
        "sentiment_records",
        ["topic", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sentiment_records_topic_created_at",
        table_name="sentiment_records",
    )
    op.drop_index(
        "ix_sentiment_records_created_at",
        table_name="sentiment_records",
    )
    op.drop_index(
        "ix_sentiment_records_sentiment_label",
        table_name="sentiment_records",
    )
    op.drop_index(
        "ix_sentiment_records_topic",
        table_name="sentiment_records",
    )
    op.drop_table("sentiment_records")
