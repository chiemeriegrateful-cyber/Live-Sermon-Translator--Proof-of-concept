from route import route, classify

with open("data/transcripts/sermon_reference.txt", "r", encoding="utf-8") as f:
    segments = [line.strip() for line in f if line.strip()]

opus_count = 0
llm_count = 0
previous_classification = None

for idx, seg in enumerate(segments):
    decision = route(seg, previous_classification)
    previous_classification = classify(seg)

    if decision == "opus":
        opus_count += 1
    else:
        llm_count += 1
    preview = seg[:80] + ("..." if len(seg) > 80 else "")
    print(f"[{idx:2d}] {decision:4s} ({len(seg.split()):3d}w) {preview}")

print(f"\nTotal: {opus_count} opus, {llm_count} llm")