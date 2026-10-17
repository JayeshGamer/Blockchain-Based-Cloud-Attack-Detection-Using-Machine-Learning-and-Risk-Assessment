"""
Unit tests for blockchain cryptographic hashing and mock ledger verification.
"""

import pytest
from src.blockchain.adapter import BlockchainAuditAdapter, compute_evidence_hash


def test_evidence_hash_deterministic():
    payload1 = {"eventId": 1, "risk": 0.85, "action": "RATE_LIMIT"}
    payload2 = {"action": "RATE_LIMIT", "eventId": 1, "risk": 0.85}  # reordered keys

    h1 = compute_evidence_hash(payload1)
    h2 = compute_evidence_hash(payload2)

    # Invariant to key insertion order
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string


def test_mock_ledger_logging_and_tamper_verification():
    adapter = BlockchainAuditAdapter(enabled=True, use_mock=True)

    evidence = {"risk_score": 0.88, "anomaly_score": 0.75, "node_id": "node_05"}
    res = adapter.log_defense_event(
        event_id=101,
        evidence_payload=evidence,
        entity_id="node_05",
        risk_category=2,
        action="TRAFFIC_FILTER_ISOLATE",
    )

    assert res["status"] == "RECORDED"
    assert res["ledger_type"] == "MOCK_LEDGER"

    # Integrity verification
    is_valid = adapter.verify_integrity(101, evidence)
    assert is_valid is True

    # Tampered payload must fail
    tampered = {"risk_score": 0.20, "anomaly_score": 0.75, "node_id": "node_05"}
    is_tampered_valid = adapter.verify_integrity(101, tampered)
    assert is_tampered_valid is False
