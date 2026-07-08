#!/usr/bin/env python3
"""Deterministic rubric-coverage evaluator for the pointdit-rk3588 methodology report.

Contract:
  - Reads .omc/research/pointdit-rk3588/REPORT.md (override with $REPORT_PATH).
  - Checks each rubric section for: anchor heading, minimum word count, minimum citations.
  - Prints a JSON scorecard to stdout.
  - Exit 0 (pass) iff coverage >= 0.85 AND no critical section (R2, R4) missing.
  - Exit 1 otherwise. Exit 2 on missing report file.

Citations counted per section = markdown links + bare URLs + arXiv ids + numeric-with-unit tokens
(so a section must show *evidence*, not just prose).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.environ.get("REPORT_PATH", os.path.join(HERE, "REPORT.md"))

# section_id -> (anchor_regex, min_words, min_citations, critical)
SECTIONS = {
    "R1": (r"^#{1,6}\s*R1\b", 250, 3, False),
    "R2": (r"^#{1,6}\s*R2\b", 250, 3, True),
    "R3": (r"^#{1,6}\s*R3\b", 180, 2, False),
    "R4": (r"^#{1,6}\s*R4\b", 350, 4, True),
    "R5": (r"^#{1,6}\s*R5\b", 180, 2, False),
}

CITATION_PATTERNS = [
    r"\[[^\]]+\]\([^)]+\)",          # markdown link
    r"https?://\S+",                  # bare url
    r"arXiv:\s*\d{4}\.\d{4,5}",       # arxiv id
    r"\b\d+(\.\d+)?\s?(GMAC|MAC|GFLOP|FLOP|TOPS|FPS|ms|GB|MB|%)\b",  # numeric evidence
]
CITE_RE = re.compile("|".join(CITATION_PATTERNS), re.IGNORECASE)


def split_sections(text):
    """Return dict section_id -> body text (from its anchor to the next R# anchor or EOF)."""
    lines = text.splitlines()
    # find anchor line indices
    anchors = []
    for sid, (rx, *_ ) in SECTIONS.items():
        pat = re.compile(rx, re.IGNORECASE | re.MULTILINE)
        idx = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
        if idx is not None:
            anchors.append((idx, sid))
    anchors.sort()
    bodies = {}
    for n, (idx, sid) in enumerate(anchors):
        end = anchors[n + 1][0] if n + 1 < len(anchors) else len(lines)
        bodies[sid] = "\n".join(lines[idx:end])
    return bodies


def evaluate():
    if not os.path.exists(REPORT_PATH):
        print(json.dumps({"error": f"report not found: {REPORT_PATH}",
                          "hint": "autoresearch가 REPORT.md를 생성/개선해야 합니다."},
                         ensure_ascii=False, indent=2))
        return 2

    text = open(REPORT_PATH, encoding="utf-8").read()
    bodies = split_sections(text)

    results = {}
    covered = 0
    critical_missing = []
    for sid, cfg in SECTIONS.items():
        min_words, min_cites, critical = cfg[1], cfg[2], cfg[3]
        body = bodies.get(sid, "")
        words = len(body.split())
        cites = len(CITE_RE.findall(body))
        present = bool(body)
        ok = present and words >= min_words and cites >= min_cites
        if ok:
            covered += 1
        elif critical:
            critical_missing.append(sid)
        results[sid] = {
            "present": present, "words": words, "min_words": min_words,
            "citations": cites, "min_citations": min_cites,
            "critical": critical, "pass": ok,
        }

    coverage = round(covered / len(SECTIONS), 3)
    passed = coverage >= 0.85 and not critical_missing

    scorecard = {
        "report": REPORT_PATH,
        "coverage": coverage,
        "threshold": 0.85,
        "critical_missing": critical_missing,
        "passed": passed,
        "sections": results,
    }
    if os.environ.get("EVAL_LLM_JUDGE") == "1":
        scorecard["note"] = "LLM-judge 보조 점수는 autoresearch 러너가 별도 주입(미구현 시 결정적 검사만 반영)."
    print(json.dumps(scorecard, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(evaluate())
