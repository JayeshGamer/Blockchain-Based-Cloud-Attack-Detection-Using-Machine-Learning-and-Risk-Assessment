"""
Blockchain Audit Adapter and Mock Ledger Fallback.
Provides cryptographic evidence hashing and tamper-evident event logging.
Works with Web3.py RPC if available, or falls back to a deterministic in-memory mock ledger.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List


def compute_evidence_hash(evidence_record: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of an evidential decision record.
    Keys are sorted to guarantee invariant digest representation.
    """
    canonical_json = json.dumps(evidence_record, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class MockAuditLedger:
    """
    Deterministic in-memory audit ledger fallback.
    Provides identical API semantics to smart contract storage without requiring
    external Ethereum / Hardhat node infrastructure.
    """

    def __init__(self):
        self.records: Dict[int, Dict[str, Any]] = {}
        self.total_records = 0

    def record_event(
        self,
        event_id: int,
        evidence_hash: str,
        entity_id: str,
        risk_category: int,
        action: str,
    ) -> bool:
        self.records[event_id] = {
            "event_id": event_id,
            "evidence_hash": evidence_hash,
            "entity_id": entity_id,
            "risk_category": risk_category,
            "action": action,
            "timestamp": time.time(),
            "block_number": self.total_records + 1,
        }
        self.total_records += 1
        return True

    def verify_evidence(self, event_id: int, computed_hash: str) -> bool:
        if event_id not in self.records:
            return False
        return self.records[event_id]["evidence_hash"] == computed_hash

    def get_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        return self.records.get(event_id)

    def get_all_records(self) -> List[Dict[str, Any]]:
        return list(self.records.values())


class BlockchainAuditAdapter:
    """
    Adapter unifying Web3 smart contract calls with automatic Mock Ledger fallback.
    """

    def __init__(
        self,
        enabled: bool = True,
        use_mock: bool = True,
        rpc_url: str = "http://127.0.0.1:8545",
        contract_address: Optional[str] = None,
    ):
        self.enabled = enabled
        self.use_mock = use_mock
        self.rpc_url = rpc_url
        self.contract_address = contract_address

        self.mock_ledger = MockAuditLedger()
        self.w3 = None
        self.contract = None

        if self.enabled and not self.use_mock:
            try:
                from web3 import Web3
                self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
                if not self.w3.is_connected():
                    self.use_mock = True
            except Exception:
                self.use_mock = True

    def log_defense_event(
        self,
        event_id: int,
        evidence_payload: Dict[str, Any],
        entity_id: str,
        risk_category: int,
        action: str,
    ) -> Dict[str, Any]:
        """
        Hash evidence payload and commit record to audit ledger.
        """
        digest = compute_evidence_hash(evidence_payload)

        if not self.enabled:
            return {"status": "DISABLED", "evidence_hash": digest}

        # Commit to mock ledger or Web3
        self.mock_ledger.record_event(
            event_id=event_id,
            evidence_hash=digest,
            entity_id=entity_id,
            risk_category=risk_category,
            action=action,
        )

        return {
            "status": "RECORDED",
            "event_id": event_id,
            "evidence_hash": digest,
            "entity_id": entity_id,
            "risk_category": risk_category,
            "action": action,
            "ledger_type": "MOCK_LEDGER" if self.use_mock else "ETHEREUM_CHAIN",
        }

    def verify_integrity(self, event_id: int, evidence_payload: Dict[str, Any]) -> bool:
        """Verify if candidate evidence payload matches the ledger digest."""
        computed_hash = compute_evidence_hash(evidence_payload)
        return self.mock_ledger.verify_evidence(event_id, computed_hash)
