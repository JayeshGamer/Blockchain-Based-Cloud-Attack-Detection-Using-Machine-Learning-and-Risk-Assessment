const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("RiskAuditLedger", function () {
  let ledger;
  let owner;
  let nodeAddr;

  beforeEach(async function () {
    [owner, nodeAddr] = await ethers.getSigners();
    const RiskAuditLedger = await ethers.getContractFactory("RiskAuditLedger");
    ledger = await RiskAuditLedger.deploy();
  });

  it("Should record and verify an immutable attack defense event", async function () {
    const eventId = 1;
    const testHash = ethers.keccak256(ethers.toUtf8Bytes("TestEvidencePayload123"));
    const entityId = "node_04";
    const riskCategory = 2; // High
    const action = "TRAFFIC_FILTER_ISOLATE";

    const tx = await ledger.recordEvent(
      eventId,
      testHash,
      entityId,
      riskCategory,
      action
    );
    await tx.wait();

    expect(await ledger.totalRecords()).to.equal(1);

    const isValid = await ledger.verifyEvidence(eventId, testHash);
    expect(isValid).to.equal(true);

    const wrongHash = ethers.keccak256(ethers.toUtf8Bytes("TamperedPayload"));
    const isTampered = await ledger.verifyEvidence(eventId, wrongHash);
    expect(isTampered).to.equal(false);
  });
});
