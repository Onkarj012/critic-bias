from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.base import Base

class Run(Base):
    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)

    condition = Column(String, nullable=False)  # source_visible | source_blind

    seed = Column(Integer, nullable=False)
    temperature_creator = Column(Float, nullable=False)
    temperature_critic = Column(Float, nullable=False)

    status = Column(
        String,
        nullable=False,
        default="created",  # created | running | completed | failed
    )

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # relationships
    prompts = relationship("Prompt", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="run", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)

    system_prompt = Column(Text, nullable=False)
    user_prompt = Column(Text, nullable=False)

    constraints = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    prompts = relationship("Prompt", back_populates="task")

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=False)

    creator_provider = Column(String, nullable=False)   # openai, anthropic
    creator_model = Column(String, nullable=False)      # gpt-4o, claude-3.5-sonnet
    creator_version = Column(String, nullable=True)

    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    run = relationship("Run", back_populates="prompts")
    task = relationship("Task", back_populates="prompts")
    critiques = relationship(
        "Critique",
        back_populates="prompt",
        cascade="all, delete-orphan",
    )

class Critique(Base):
    __tablename__ = "critiques"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    prompt_id = Column(UUID(as_uuid=True), ForeignKey("prompts.id"), nullable=False)

    critic_provider = Column(String, nullable=False)
    critic_model = Column(String, nullable=False)
    critic_version = Column(String, nullable=True)

    source_visible = Column(Boolean, nullable=False)

    score = Column(Float, nullable=False)

    strengths = Column(JSONB, nullable=False)    # list[str]
    weaknesses = Column(JSONB, nullable=False)   # list[str]
    suggestions = Column(JSONB, nullable=False)  # list[str]

    tone = Column(String, nullable=True)  # polite | neutral | brutal

    raw_text = Column(Text, nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    prompt = relationship("Prompt", back_populates="critiques")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False)

    name = Column(String, nullable=False)         # MFI, BI, CR, TPS, BPS
    target_model = Column(String, nullable=False) # model identifier

    value = Column(Float, nullable=False)
    meta_data = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    run = relationship("Run", back_populates="metrics")
