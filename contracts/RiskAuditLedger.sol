// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title RiskAuditLedger
 * @dev Lightweight tamper-evident audit ledger for cloud attack tolerance events.
 * Records only cryptographic hashes of evidence records and metadata.
 * Privacy-preserving: Raw packet flows and sensitive PII are never committed on-chain.
 */
contract RiskAuditLedger {
    struct AuditRecord {
        uint256 eventId;
        bytes32 evidenceHash;   // SHA-256 digest of {evidence, risk_scores, timestamp}
        string entityId;        // Affected node or cluster identifier
        uint8 riskCategory;     // 0: Low, 1: Medium, 2: High, 3: Critical
        string action;          // Defense action (e.g., RATE_LIMIT, ISOLATE)
        uint256 blockTimestamp;
        address recordedBy;
    }

    address public owner;
    uint256 public totalRecords;
    mapping(uint256 => AuditRecord) public records;

    event RiskEventLogged(
        uint256 indexed eventId,
        bytes32 indexed evidenceHash,
        string entityId,
        uint8 riskCategory,
        string action,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only contract owner can execute this");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalRecords = 0;
    }

    /**
     * @dev Record a new defensive audit event with evidence hash.
     */
    function recordEvent(
        uint256 _eventId,
        bytes32 _evidenceHash,
        string calldata _entityId,
        uint8 _riskCategory,
        string calldata _action
    ) external returns (bool) {
        records[_eventId] = AuditRecord({
            eventId: _eventId,
            evidenceHash: _evidenceHash,
            entityId: _entityId,
            riskCategory: _riskCategory,
            action: _action,
            blockTimestamp: block.timestamp,
            recordedBy: msg.sender
        });

        totalRecords++;

        emit RiskEventLogged(
            _eventId,
            _evidenceHash,
            _entityId,
            _riskCategory,
            _action,
            block.timestamp
        );

        return true;
    }

    /**
     * @dev Verify if a local evidence record hash matches the on-chain recorded digest.
     */
    function verifyEvidence(uint256 _eventId, bytes32 _computedHash) external view returns (bool) {
        require(_eventId <= totalRecords && _eventId > 0, "Event ID does not exist");
        return records[_eventId].evidenceHash == _computedHash;
    }

    /**
     * @dev Retrieve recorded event metadata.
     */
    function getEvent(uint256 _eventId) external view returns (
        bytes32 evidenceHash,
        string memory entityId,
        uint8 riskCategory,
        string memory action,
        uint256 blockTimestamp,
        address recordedBy
    ) {
        AuditRecord memory rec = records[_eventId];
        return (
            rec.evidenceHash,
            rec.entityId,
            rec.riskCategory,
            rec.action,
            rec.blockTimestamp,
            rec.recordedBy
        );
    }
}
