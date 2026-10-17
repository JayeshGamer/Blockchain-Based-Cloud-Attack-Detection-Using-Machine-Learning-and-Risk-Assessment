"""
Traffic Injection and Simulation Script for Live Viva Demonstrations.
Generates realistic benign HTTP traffic and simulated attack bursts
(SYN Flood, PortScan, Volumetric Floods) and feeds them into the LiveFlowSniffer
and FastAPI prediction endpoint, verifying live detection and defensive triggers.
"""

import sys
import os
import time
import json
import random
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.capture.sniffer import LiveFlowSniffer


def simulate_benign_http(sniffer: LiveFlowSniffer, src_ip: str = "192.168.1.105") -> Dict[str, Any]:
    """Simulates a standard benign HTTP GET interaction with normal round-trip IATs."""
    t0 = time.time()
    src_port = random.randint(49152, 65535)
    dst_ip = "10.0.0.50"
    dst_port = 80

    # 1. TCP SYN
    sniffer.process_packet(src_ip, src_port, dst_ip, dst_port, 6, pkt_len=0, flags={"SYN": 1}, timestamp=t0)
    # 2. TCP SYN-ACK
    sniffer.process_packet(dst_ip, dst_port, src_ip, src_port, 6, pkt_len=0, flags={"SYN": 1, "ACK": 1}, timestamp=t0 + 0.015)
    # 3. TCP ACK
    sniffer.process_packet(src_ip, src_port, dst_ip, dst_port, 6, pkt_len=0, flags={"ACK": 1}, timestamp=t0 + 0.018)
    # 4. HTTP GET
    sniffer.process_packet(src_ip, src_port, dst_ip, dst_port, 6, pkt_len=450, flags={"PSH": 1, "ACK": 1}, timestamp=t0 + 0.022)
    # 5. HTTP 200 OK + Payload
    sniffer.process_packet(dst_ip, dst_port, src_ip, src_port, 6, pkt_len=1460, flags={"ACK": 1}, timestamp=t0 + 0.045)
    sniffer.process_packet(dst_ip, dst_port, src_ip, src_port, 6, pkt_len=1460, flags={"ACK": 1}, timestamp=t0 + 0.048)
    # 6. TCP FIN-ACK handshake (closing connection)
    record = sniffer.process_packet(src_ip, src_port, dst_ip, dst_port, 6, pkt_len=0, flags={"FIN": 1, "ACK": 1}, timestamp=t0 + 0.075)
    return record


def simulate_syn_flood(sniffer: LiveFlowSniffer, attacker_ip: str = "172.16.0.99", num_packets: int = 40) -> Dict[str, Any]:
    """Simulates a high-rate SYN flood burst (zero ACKs, tiny packets, high packet rate)."""
    t0 = time.time()
    src_port = random.randint(30000, 40000)
    dst_ip = "10.0.0.50"
    dst_port = 80

    for i in range(num_packets):
        sniffer.process_packet(
            attacker_ip, src_port, dst_ip, dst_port, 6,
            pkt_len=0, flags={"SYN": 1},
            timestamp=t0 + (i * 0.0005)  # 2,000 pkts/sec burst
        )
    # Force timeout expiration to extract flow
    records = sniffer.flush_expired(current_time=t0 + 10.0)
    return records[0] if records else None


def simulate_port_scan(sniffer: LiveFlowSniffer, scanner_ip: str = "10.0.1.200", ports: list = None) -> list:
    """Simulates rapid probe scans across multiple ports."""
    ports = ports or [21, 22, 23, 25, 80, 443, 3389, 8080]
    t0 = time.time()
    records = []
    for i, port in enumerate(ports):
        src_port = 50000 + i
        # Send probe packet with RST response
        sniffer.process_packet(scanner_ip, src_port, "10.0.0.50", port, 6, pkt_len=0, flags={"SYN": 1}, timestamp=t0 + (i * 0.002))
        rec = sniffer.process_packet("10.0.0.50", port, scanner_ip, src_port, 6, pkt_len=0, flags={"RST": 1}, timestamp=t0 + (i * 0.002) + 0.001)
        if rec:
            records.append(rec)
    return records


if __name__ == "__main__":
    print("=" * 70)
    print("LIVE NETWORK FLOW SIMULATION & PREDICTION DEMO")
    print("=" * 70)

    sniffer = LiveFlowSniffer()

    print("\n[1] Simulating Benign HTTP Web Browsing...")
    benign_flow = simulate_benign_http(sniffer)
    if benign_flow:
        print(f" -> Assembled Flow: {benign_flow['flow_id']}")
        print(f" -> Duration: {benign_flow['duration']*1000:.2f} ms | Packets: {benign_flow['packets']}")
        print(f" -> Generated Features: {len(benign_flow['features'])} numerical values")

    print("\n[2] Simulating Rapid High-Rate SYN Flood Burst...")
    attack_flow = simulate_syn_flood(sniffer)
    if attack_flow:
        print(f" -> Assembled Flow: {attack_flow['flow_id']}")
        print(f" -> Packets: {attack_flow['packets']} | Duration: {attack_flow['duration']*1000:.2f} ms")

    # Test against local API if running
    # Test against local API if running
    api_url = "http://127.0.0.1:8000/predict"
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health")
        with urllib.request.urlopen(req, timeout=1.0) as res:
            if res.status == 200:
                print("\n[3] Testing Live Inference Against Running FastAPI Backend...")
                for label, flow in [("BENIGN HTTP", benign_flow), ("SYN FLOOD ATTACK", attack_flow)]:
                    if not flow: continue
                    post_data = json.dumps({"features": flow["features"]}).encode("utf-8")
                    p_req = urllib.request.Request(api_url, data=post_data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(p_req, timeout=2.0) as p_res:
                        data = json.loads(p_res.read().decode("utf-8"))
                        print(f" [{label}] -> Prediction: {data['prediction']} | Attack Score: {data['attack_score']:.4f}")
    except Exception:
        print("\nNote: FastAPI backend not running on port 8000. Start it via 'uvicorn src.api.main:app' to test live API calls.")
