# Blockchain Audit Ledger Module

## Overview
This module provides a lightweight, tamper-evident audit ledger (`RiskAuditLedger.sol`) for recording defensive cloud actions and SHA-256 evidence record hashes.

> **Scope Note:** This Solidity contract is designed as an optional audit module. It does not attempt to reproduce the full multi-organization Hyperledger Fabric and Proof of Risk consensus mechanism from the base paper, which requires large-scale containerized infrastructure. Instead, it provides a cryptographically verifiable provenance trail of risk events and defensive interventions.

## Privacy Preservation
Raw network flows, IP addresses, and sensitive user payloads are **never committed on-chain**. Only:
1. `eventId`: Unique numerical identifier
2. `evidenceHash`: SHA-256 digest of the evidential decision record
3. `entityId`: Node or cluster identifier (e.g. `node_03`)
4. `riskCategory`: Severity tier (0: Low, 1: Medium, 2: High, 3: Critical)
5. `action`: Defense action triggered (e.g. `RATE_LIMIT`, `TRAFFIC_FILTER_ISOLATE`)
6. `blockTimestamp`: Block mining timestamp

## Testing with Hardhat
```bash
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
npx hardhat test contracts/test/RiskAuditLedger.test.js
```

## Python Integration
The Python backend uses `src/blockchain/adapter.py`, which features:
1. Direct Web3.py RPC client if an Ethereum-compatible RPC (e.g. Hardhat / Anvil / Ganache) is running.
2. An automatic in-memory SQLite mock ledger fallback, allowing the entire project to run seamlessly without blockchain node dependencies.
