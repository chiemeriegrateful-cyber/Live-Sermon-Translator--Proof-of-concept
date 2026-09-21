"""Fan-out load test for the /ws subtitle endpoint.

Opens N concurrent clients, holds them for a duration, and reports connect
failures, dropped messages (sequence gaps) and delivery latency (client receive
time minus the server's timestamp_ms, so run it on the same machine as the
server or accept clock skew).

--slow K makes K of the clients deliberately slow readers (sleeping
--slow-delay seconds after every message) and reports them separately. It
checks isolation: healthy clients must stay clean while slow ones lag.
It does not reach the broker's drop-oldest path: the client library and kernel
socket buffers absorb the backlog first, so slow clients show lag, not gaps.
The drop-oldest logic itself is checked by publishing into an undrained
Broker subscriber directly.

    python scripts/load_ws.py --url ws://localhost:8001/ws --clients 100 --duration 60
    python scripts/load_ws.py --clients 300 --slow 10 --slow-delay 0.5
"""
import argparse
import asyncio
import json
import statistics
import time

import websockets

# Messages in the first moments after connecting are the ring-buffer catch-up,
# not live traffic, so they are excluded from latency.
CATCHUP_WINDOW_S = 1.0


def new_stats():
    return {"connected": 0, "failed": 0, "gapped_clients": 0, "gaps": 0,
            "counts": [], "latencies_ms": [], "errors": []}


async def client(url: str, duration: float, stats: dict, idx: int,
                 slow_delay: float = 0.0):
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            connected_at = time.monotonic()
            stats["connected"] += 1
            count = 0
            gaps = 0
            last_seq = None
            deadline = connected_at + duration
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                now = time.monotonic()
                count += 1
                msg = json.loads(raw)
                # Seq is consecutive for a client that lost nothing (catch-up
                # then live), so a jump means the broker dropped messages.
                seq = msg.get("seq")
                if last_seq is not None and seq != last_seq + 1:
                    gaps += 1
                last_seq = seq
                if now - connected_at > CATCHUP_WINDOW_S:
                    sent_ms = msg.get("timestamp_ms")
                    if sent_ms:
                        stats["latencies_ms"].append(time.time() * 1000 - sent_ms)
                if slow_delay:
                    await asyncio.sleep(slow_delay)
            stats["counts"].append(count)
            stats["gaps"] += gaps
            stats["gapped_clients"] += 1 if gaps else 0
    except Exception as e:
        stats["failed"] += 1
        stats["errors"].append(f"client {idx}: {e!r}")


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def report(label: str, total: int, stats: dict):
    print(f"[{label}] connected {stats['connected']}/{total}, failed {stats['failed']}")
    if stats["counts"]:
        c = stats["counts"]
        print(f"  messages per client: min {min(c)} median {statistics.median(c)} max {max(c)}")
        print(f"  clients with dropped messages: {stats['gapped_clients']}/{len(c)} "
              f"(total gaps {stats['gaps']})")
    lat = stats["latencies_ms"]
    if lat:
        print(f"  live latency ms: p50 {pct(lat, 0.5):.0f} p95 {pct(lat, 0.95):.0f} "
              f"max {max(lat):.0f} (n={len(lat)})")
    else:
        print("  no live messages after the catch-up window (was audio flowing?)")
    for err in stats["errors"][:3]:
        print("  ", err)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8001/ws")
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--slow", type=int, default=0,
                    help="how many of the clients read slowly")
    ap.add_argument("--slow-delay", type=float, default=0.5,
                    help="seconds a slow client sleeps after each message")
    args = ap.parse_args()

    slow_n = min(args.slow, args.clients)
    healthy, slow = new_stats(), new_stats()
    print(f"{args.clients} clients ({slow_n} slow) -> {args.url} for {args.duration:.0f}s")
    await asyncio.gather(
        *(
            client(args.url, args.duration,
                   slow if i < slow_n else healthy, i,
                   args.slow_delay if i < slow_n else 0.0)
            for i in range(args.clients)
        )
    )
    report("healthy", args.clients - slow_n, healthy)
    if slow_n:
        report("slow", slow_n, slow)


if __name__ == "__main__":
    asyncio.run(main())
