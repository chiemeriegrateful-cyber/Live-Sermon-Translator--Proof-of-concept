"""Checks for pipeline/router.py routing rules. Run from the repo root:

    python scripts/test_pipeline_router.py
"""
import sys

sys.path.insert(0, ".")
from pipeline.router import route  # noqa: E402


def run(segments):
    state, out = {}, []
    for seg in segments:
        decision, state = route(seg, state)
        out.append(decision)
    return out


# The failing stretch from the Mark 11 run: punchy fragments after a sentence.
punchy = [
    "Was traf Jesus in Jerusalem an?",   # 6 words, first short -> opus
    "Da war viel Volk.",                 # short run of 2 -> llm
    "Ein großes Getümmel!",              # llm
    "Zivilbeschäftigung!",               # llm, sees the two before as context
    "Manche vielleicht nicht.",          # llm
]
assert run(punchy) == ["opus", "llm", "llm", "llm", "llm"], run(punchy)

# A single short line between full sentences stays on opus.
isolated = [
    "Heute möchte ich mit euch über Hingabe sprechen und was sie bedeutet.",
    "Amen.",
    "Und deshalb ist es wichtig, dass wir verstehen, was Gott von uns möchte.",
]
assert run(isolated) == ["opus", "opus", "opus"], run(isolated)

# A long segment resets the run, so the next short one is not a "run of 2".
reset = [
    "Da war viel Volk.",
    "Und deshalb ist es wichtig, dass wir verstehen, was Gott von uns möchte, "
    "wenn wir ihm wirklich nachfolgen wollen in jedem Bereich unseres Lebens hier.",
    "Amen.",
]
assert run(reset) == ["opus", "opus", "opus"], run(reset)

# Tail of a clipped sentence starts lowercase and needs context.
assert run(["Er kam, um ein Opfer zu bringen und dann ging er wieder weg von dort.",
            "um ein Opfer zu bringen."]) == ["opus", "llm"]

# Scripture cue behaviour is unchanged.
assert run(["Wir lesen aus Markus 11."])[0] == "llm"

print("router checks passed")
