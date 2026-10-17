"""
Unit tests for data cleaning and label normalization.
"""

import pytest
from src.data.audit import normalize_label, ATTACK_FAMILY_MAPPING


def test_normalize_benign():
    assert normalize_label("BENIGN") == "BENIGN"
    assert normalize_label(" BENIGN ") == "BENIGN"


def test_normalize_web_attack_unicode():
    # Corrupted encoding with replacement character \ufffd
    assert normalize_label("Web Attack \ufffd Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("Web Attack \ufffd XSS") == "Web Attack - XSS"
    assert normalize_label("Web Attack \ufffd Sql Injection") == "Web Attack - Sql Injection"


def test_normalize_whitespace_and_cases():
    assert normalize_label("  DDoS  ") == "DDoS"
    assert normalize_label("PortScan") == "PortScan"
    assert normalize_label("DoS Hulk") == "DoS Hulk"
    assert normalize_label("Heartbleed") == "Heartbleed"


def test_attack_family_mapping():
    assert ATTACK_FAMILY_MAPPING["BENIGN"] == "BENIGN"
    assert ATTACK_FAMILY_MAPPING["DDoS"] == "DDoS"
    assert ATTACK_FAMILY_MAPPING["PortScan"] == "PortScan"
    assert ATTACK_FAMILY_MAPPING["DoS Hulk"] == "DoS"
    assert ATTACK_FAMILY_MAPPING["Web Attack - XSS"] == "Web Attack"
    assert ATTACK_FAMILY_MAPPING["FTP-Patator"] == "Brute Force"
    assert ATTACK_FAMILY_MAPPING["Heartbleed"] == "Exploit/Heartbleed"
