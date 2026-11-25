"""
QuantGuard-MAS — Unit IV: FIPA-ACL Message Bus, Shared Blackboard & Contract Net Protocol
"""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import Config


VALID_PERFORMATIVES = {
    "cfp",
    "propose",
    "accept-proposal",
    "reject-proposal",
    "inform",
    "request",
    "refuse",
    "agree",
}


@dataclass
class FIPAMessage:
    performative: str
    sender: str
    receiver: str
    ontology: str
    content: Dict[str, Any]
    conversation_id: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if self.performative not in VALID_PERFORMATIVES:
            raise ValueError(f"Invalid FIPA performative: {self.performative}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SharedBlackboard:
    """
    Unit IV — Shared Blackboard + SQLite FIPA-ACL Audit Logger.
    All 4 agents read/write structured domain knowledge and FIPA envelopes here.
    """

    def __init__(self, run_id: Optional[str] = None, db_path: Optional[Path] = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.db_path = Path(db_path or Config.SQLITE_DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.messages: List[FIPAMessage] = []
        self.state: Dict[str, Any] = {
            "run_id": self.run_id,
            "goal": "",
            "market_snapshot": {},
            "rag_memories": [],
            "finbert_sentiments": [],
            "tot_branches": [],
            "quant_report": {},
            "audit_verdict": {},
            "conflict_arbitration": {},
            "final_allocation": {},
            "hitl_status": "PENDING",
            "hitl_feedback": None,
        }
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fipa_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    performative TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    receiver TEXT NOT NULL,
                    ontology TEXT NOT NULL,
                    content_json TEXT NOT NULL
                )
                """
            )

    def post(
        self,
        performative: str,
        sender: str,
        receiver: str,
        ontology: str,
        content: Dict[str, Any],
    ) -> FIPAMessage:
        msg = FIPAMessage(
            performative=performative,
            sender=sender,
            receiver=receiver,
            ontology=ontology,
            content=content,
            conversation_id=self.run_id,
        )
        self.messages.append(msg)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO fipa_messages (
                    run_id, message_id, timestamp, performative,
                    sender, receiver, ontology, content_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    msg.message_id,
                    msg.timestamp,
                    msg.performative,
                    msg.sender,
                    msg.receiver,
                    msg.ontology,
                    json.dumps(msg.content, ensure_ascii=False),
                ),
            )
        return msg

    def get_messages(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    @classmethod
    def fetch_logged_messages(cls, run_id: str, db_path: Optional[Path] = None) -> List[Dict]:
        path = Path(db_path or Config.SQLITE_DATABASE_PATH)
        if not path.exists():
            return []
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM fipa_messages WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["content"] = json.loads(d.pop("content_json"))
            out.append(d)
        return out


class ContractNetBroker:
    """
    Unit IV — Contract Net Protocol (CFP -> Propose -> Accept/Reject).
    Orchestrator broadcasts task calls and awards execution to best-bidding agents.
    """

    def __init__(self, blackboard: SharedBlackboard):
        self.bb = blackboard

    def run_auction(self, task_name: str, bidders: List[Dict[str, Any]]) -> str:
        """
        bidders: [{"agent": "ResearcherAgent", "confidence": 0.92, "est_latency_ms": 450}, ...]
        Returns the winning agent name after logging full CFP/Propose/Accept sequence.
        """
        self.bb.post(
            performative="cfp",
            sender="OrchestratorAgent",
            receiver="ALL_AGENTS",
            ontology="task_allocation",
            content={"task": task_name, "eligible_bidders": [b["agent"] for b in bidders]},
        )

        for bid in bidders:
            self.bb.post(
                performative="propose",
                sender=bid["agent"],
                receiver="OrchestratorAgent",
                ontology="task_allocation",
                content={
                    "task": task_name,
                    "confidence": bid["confidence"],
                    "est_latency_ms": bid["est_latency_ms"],
                },
            )

        winner = max(bidders, key=lambda b: (b["confidence"], -b["est_latency_ms"]))
        for bid in bidders:
            perf = "accept-proposal" if bid["agent"] == winner["agent"] else "reject-proposal"
            self.bb.post(
                performative=perf,
                sender="OrchestratorAgent",
                receiver=bid["agent"],
                ontology="task_allocation",
                content={"task": task_name, "awarded": perf == "accept-proposal"},
            )
        return winner["agent"]
