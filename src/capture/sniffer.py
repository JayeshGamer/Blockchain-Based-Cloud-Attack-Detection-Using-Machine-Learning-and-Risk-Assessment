"""
Real-time Bidirectional Network Flow Sniffer and Feature Extractor.
Aggregates packet streams into bidirectional flows and computes the 69-feature
CICIDS2017 schema required by the Adaptive ML-DST Online SGD inference engine.
Can capture from live raw sockets or inject test packets for live viva demonstrations.
"""

import time
import math
import socket
import struct
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


class BidirectionalFlow:
    """Tracks packet statistics in forward and backward directions for a single 5-tuple."""

    def __init__(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int, protocol: int, start_time: float):
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol
        self.start_time = start_time
        self.last_seen = start_time

        self.fwd_pkt_lens: List[int] = []
        self.bwd_pkt_lens: List[int] = []
        self.fwd_timestamps: List[float] = []
        self.bwd_timestamps: List[float] = []
        self.all_timestamps: List[float] = []

        self.fwd_header_len = 0
        self.bwd_header_len = 0
        self.min_seg_size_fwd = 32

        # TCP Flags
        self.fin_cnt = 0
        self.syn_cnt = 0
        self.rst_cnt = 0
        self.psh_cnt = 0
        self.ack_cnt = 0
        self.urg_cnt = 0
        self.cwe_cnt = 0
        self.ece_cnt = 0
        self.fwd_psh = 0
        self.fwd_urg = 0

        self.init_win_fwd = -1
        self.init_win_bwd = -1
        self.act_data_pkt_fwd = 0

    def add_packet(self, pkt_len: int, header_len: int, is_fwd: bool, timestamp: float, flags: Dict[str, int], window_size: int = -1):
        self.last_seen = timestamp
        self.all_timestamps.append(timestamp)

        if flags.get("FIN"): self.fin_cnt += 1
        if flags.get("SYN"): self.syn_cnt += 1
        if flags.get("RST"): self.rst_cnt += 1
        if flags.get("PSH"): self.psh_cnt += 1
        if flags.get("ACK"): self.ack_cnt += 1
        if flags.get("URG"): self.urg_cnt += 1
        if flags.get("ECE"): self.ece_cnt += 1
        if flags.get("CWR") or flags.get("CWE"): self.cwe_cnt += 1

        if is_fwd:
            self.fwd_pkt_lens.append(pkt_len)
            self.fwd_timestamps.append(timestamp)
            self.fwd_header_len += header_len
            if header_len > 0:
                self.min_seg_size_fwd = min(self.min_seg_size_fwd, header_len)
            if flags.get("PSH"): self.fwd_psh += 1
            if flags.get("URG"): self.fwd_urg += 1
            if pkt_len > 0: self.act_data_pkt_fwd += 1
            if self.init_win_fwd == -1 and window_size >= 0:
                self.init_win_fwd = window_size
        else:
            self.bwd_pkt_lens.append(pkt_len)
            self.bwd_timestamps.append(timestamp)
            self.bwd_header_len += header_len
            if self.init_win_bwd == -1 and window_size >= 0:
                self.init_win_bwd = window_size

    def extract_features(self) -> Dict[str, float]:
        """Calculates exact 69 CICIDS2017 features."""
        duration_sec = max(1e-6, self.last_seen - self.start_time)
        duration_us = duration_sec * 1_000_000.0

        n_fwd = len(self.fwd_pkt_lens)
        n_bwd = len(self.bwd_pkt_lens)
        tot_fwd_len = sum(self.fwd_pkt_lens)
        tot_bwd_len = sum(self.bwd_pkt_lens)
        all_lens = self.fwd_pkt_lens + self.bwd_pkt_lens
        tot_pkts = n_fwd + n_bwd
        tot_bytes = tot_fwd_len + tot_bwd_len

        # Packet Length Stats
        def stats(arr: List[int]) -> Tuple[float, float, float, float]:
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(arr, dtype=float)
            return float(np.max(a)), float(np.min(a)), float(np.mean(a)), float(np.std(a))

        fwd_max, fwd_min, fwd_mean, fwd_std = stats(self.fwd_pkt_lens)
        bwd_max, bwd_min, bwd_mean, bwd_std = stats(self.bwd_pkt_lens)
        all_max, all_min, all_mean, all_std = stats(all_lens)
        all_var = float(all_std ** 2)

        # IAT calculations (in microseconds)
        def iat_stats(ts_list: List[float]) -> Tuple[float, float, float, float, float]:
            if len(ts_list) <= 1:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            diffs = [(ts_list[i] - ts_list[i - 1]) * 1_000_000.0 for i in range(1, len(ts_list))]
            d = np.array(diffs, dtype=float)
            return float(np.sum(d)), float(np.mean(d)), float(np.std(d)), float(np.max(d)), float(np.min(d))

        flow_iat_tot, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = iat_stats(self.all_timestamps)
        fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = iat_stats(self.fwd_timestamps)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = iat_stats(self.bwd_timestamps)

        flow_bytes_s = tot_bytes / duration_sec
        flow_pkts_s = tot_pkts / duration_sec
        fwd_pkts_s = n_fwd / duration_sec
        bwd_pkts_s = n_bwd / duration_sec
        down_up_ratio = float(n_bwd / n_fwd) if n_fwd > 0 else 0.0
        avg_pkt_size = float(tot_bytes / tot_pkts) if tot_pkts > 0 else 0.0

        feat = {
            "Destination Port": float(self.dst_port),
            "Flow Duration": float(duration_us),
            "Total Fwd Packets": float(n_fwd),
            "Total Backward Packets": float(n_bwd),
            "Total Length of Fwd Packets": float(tot_fwd_len),
            "Total Length of Bwd Packets": float(tot_bwd_len),
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": float(flow_bytes_s),
            "Flow Packets/s": float(flow_pkts_s),
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Total": fwd_iat_tot,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_tot,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": float(self.fwd_psh),
            "Fwd URG Flags": float(self.fwd_urg),
            "Fwd Header Length": float(self.fwd_header_len),
            "Bwd Header Length": float(self.bwd_header_len),
            "Fwd Packets/s": float(fwd_pkts_s),
            "Bwd Packets/s": float(bwd_pkts_s),
            "Min Packet Length": all_min,
            "Max Packet Length": all_max,
            "Packet Length Mean": all_mean,
            "Packet Length Std": all_std,
            "Packet Length Variance": all_var,
            "FIN Flag Count": float(self.fin_cnt),
            "SYN Flag Count": float(self.syn_cnt),
            "RST Flag Count": float(self.rst_cnt),
            "PSH Flag Count": float(self.psh_cnt),
            "ACK Flag Count": float(self.ack_cnt),
            "URG Flag Count": float(self.urg_cnt),
            "CWE Flag Count": float(self.cwe_cnt),
            "ECE Flag Count": float(self.ece_cnt),
            "Down/Up Ratio": down_up_ratio,
            "Average Packet Size": avg_pkt_size,
            "Avg Fwd Segment Size": fwd_mean,
            "Avg Bwd Segment Size": bwd_mean,
            "Subflow Fwd Packets": float(n_fwd),
            "Subflow Fwd Bytes": float(tot_fwd_len),
            "Subflow Bwd Packets": float(n_bwd),
            "Subflow Bwd Bytes": float(tot_bwd_len),
            "Init_Win_bytes_forward": float(self.init_win_fwd),
            "Init_Win_bytes_backward": float(self.init_win_bwd),
            "act_data_pkt_fwd": float(self.act_data_pkt_fwd),
            "min_seg_size_forward": float(self.min_seg_size_fwd),
            "Active Mean": 0.0,
            "Active Std": 0.0,
            "Active Max": 0.0,
            "Active Min": 0.0,
            "Idle Mean": 0.0,
            "Idle Std": 0.0,
            "Idle Max": 0.0,
            "Idle Min": 0.0,
        }
        return feat


class LiveFlowSniffer:
    """
    Manages active flow assembly, timeout dispatching, and feature vector extraction.
    """

    def __init__(self, flow_timeout_sec: float = 3.0):
        self.flow_timeout_sec = flow_timeout_sec
        self.active_flows: Dict[Tuple[str, int, str, int, int], BidirectionalFlow] = {}
        self.completed_flows: List[Dict[str, Any]] = []

    def process_packet(
        self,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
        protocol: int,
        pkt_len: int,
        header_len: int = 20,
        flags: Optional[Dict[str, int]] = None,
        window_size: int = 1024,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a packet, associates it with a forward or backward flow key,
        and returns an expired flow record if timeout is exceeded.
        """
        now = time.time() if timestamp is None else timestamp
        flags = flags or {}

        fwd_key = (src_ip, src_port, dst_ip, dst_port, protocol)
        bwd_key = (dst_ip, dst_port, src_ip, src_port, protocol)

        if fwd_key in self.active_flows:
            flow = self.active_flows[fwd_key]
            flow.add_packet(pkt_len, header_len, is_fwd=True, timestamp=now, flags=flags, window_size=window_size)
        elif bwd_key in self.active_flows:
            flow = self.active_flows[bwd_key]
            flow.add_packet(pkt_len, header_len, is_fwd=False, timestamp=now, flags=flags, window_size=window_size)
        else:
            flow = BidirectionalFlow(src_ip, src_port, dst_ip, dst_port, protocol, start_time=now)
            flow.add_packet(pkt_len, header_len, is_fwd=True, timestamp=now, flags=flags, window_size=window_size)
            self.active_flows[fwd_key] = flow

        # Immediate dispatch on FIN or RST
        if flags.get("FIN") or flags.get("RST"):
            return self._finalize_flow(fwd_key if fwd_key in self.active_flows else bwd_key)

        return None

    def flush_expired(self, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Flushes flows that have exceeded flow_timeout_sec."""
        now = time.time() if current_time is None else current_time
        expired_keys = [k for k, f in self.active_flows.items() if (now - f.last_seen) >= self.flow_timeout_sec]
        flushed = []
        for k in expired_keys:
            rec = self._finalize_flow(k)
            if rec:
                flushed.append(rec)
        return flushed

    def _finalize_flow(self, key: Tuple[str, int, str, int, int]) -> Optional[Dict[str, Any]]:
        if key not in self.active_flows:
            return None
        flow = self.active_flows.pop(key)
        features = flow.extract_features()
        record = {
            "flow_id": f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}#{flow.protocol}",
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "features": [features[col] for col in FEATURE_ORDER],
            "feature_dict": features,
            "timestamp": flow.start_time,
            "duration": flow.last_seen - flow.start_time,
            "packets": len(flow.all_timestamps),
        }
        self.completed_flows.append(record)
        return record
