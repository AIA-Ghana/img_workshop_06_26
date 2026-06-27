

"""
image_report_pipeline.py
Fully agentic image analysis pipeline using OpenBMB MiniCPM models.

  Stage 0 (Memory):     reads runs.log — past results inform the planner
  Stage 1 (Eyes):       MiniCPM-V-4.6  — vision extraction
  Stage 1b (Eyes+):     MiniCPM-V-4.6  — detailed re-inspection (if planner requests it)
  Stage 2 (Planner):    MiniCPM4.1-8B  — decides next action based on vision output + memory
  Stage 3 (Brain):      MiniCPM4.1-8B  — structured report generation
  Stage 4 (Critic):     MiniCPM4.1-8B  — reviews quality + assigns confidence score (1–10)
  Stage 5 (Tool Picker):MiniCPM4.1-8B  — picks tools AND writes dynamic arguments

Planner decisions:
  generate_report  → description is clear, proceed to Stage 3
  reinspect        → description is vague, re-run vision with a sharper prompt
  flag_unclear     → image is unrelated/confusing, save a notice and exit

Critic output:
  verdict     → approved / regenerate
  confidence  → 1–10 score; below CONFIDENCE_THRESHOLD → flag_for_human_review tool is added
  feedback    → one sentence used to improve the next regeneration attempt

Tool Picker — picks tools AND generates their arguments:
  save_to_db           → always runs; logs report + confidence to reports_db.jsonl
  send_alert           → agent writes the alert message itself
  crop_and_reanalyze   → agent specifies which region/detail to zoom into
  flag_for_human_review→ added automatically when confidence < CONFIDENCE_THRESHOLD

Memory:
  runs.log  → one JSON line per run; planner reads last N_MEMORY_RUNS entries before deciding

Usage:
  python image_report_pipeline.py <image_path> [output.txt]
"""

import requests
import base64
import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# Auth & endpoints
# ─────────────────────────────────────────────

def get_auth() -> dict:
    """Return shared request headers. Swap os.getenv for a hardcoded token if needed."""
    token = os.getenv("MINICPM_TOKEN", "sk-minicpm-V8bcD-YTAMxECagaKOnbwTCN69IlN2LhSezGOgq2Ues")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

VISION_URL   = "http://35.203.155.71:8003/v1/chat/completions"  # MiniCPM-V-4.6
LANGUAGE_URL = "http://35.203.155.71:8001/v1/chat/completions"  # MiniCPM4.1-8B

MAX_RETRIES          = 2    # max times the Critic can send the report back for regeneration
CONFIDENCE_THRESHOLD = 6    # scores below this trigger flag_for_human_review
N_MEMORY_RUNS        = 5    # how many past runs the Planner reads from runs.log
MEMORY_LOG           = Path("runs.log")  # one JSON line per completed run


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_api(url: str, payload: dict) -> str:
    """Generic POST to any OpenAI-compatible endpoint. Raises on HTTP error."""
    resp = requests.post(url, headers=get_auth(), json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def save_report(report: str, output_path: str) -> None:
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\n[Saved] Report written to: {output_path}")


# ─────────────────────────────────────────────
# Memory — runs.log read/write
# ─────────────────────────────────────────────

def load_memory() -> list[dict]:
    """Returns the last N_MEMORY_RUNS entries from runs.log, oldest first.
    Returns an empty list if the log doesn't exist yet."""
    if not MEMORY_LOG.exists():
        return []
    lines = MEMORY_LOG.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-N_MEMORY_RUNS:]
    records = []
    for line in recent:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def save_memory(image_path: str, planner_decision: str,
                confidence: int, verdict: str, tools_called: list[str]) -> None:
    """Appends a compact run summary to runs.log."""
    record = {
        "ts":         datetime.now().isoformat(),
        "image":      image_path,
        "decision":   planner_decision,
        "confidence": confidence,
        "verdict":    verdict,
        "tools":      tools_called,
    }
    with MEMORY_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  [Memory] Run logged to {MEMORY_LOG}")


# ─────────────────────────────────────────────
# Stage 1 — Eyes: Vision extraction
# ─────────────────────────────────────────────

def extract_image_info(image_path: str) -> str:
    print(f"[Stage 1] Extracting info from image: {image_path}")
    b64 = encode_image(image_path)

    payload = {
        "model": "MiniCPM-V-4.6",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    "Describe this image in detail. "
                    "List all objects, text, people, colors, and any notable context you observe."
                )},
            ],
        }],
        "max_tokens": 500,
    }

    return call_api(VISION_URL, payload)


# ─────────────────────────────────────────────
# Stage 1b — Eyes+: Detailed re-inspection
# ─────────────────────────────────────────────

def extract_image_info_detailed(image_path: str) -> str:
    """Re-runs vision extraction with a more targeted prompt when the
    first pass was too vague. Called only if the Planner says 'reinspect'."""
    print(f"[Stage 1b] Re-inspecting image with detailed prompt: {image_path}")
    b64 = encode_image(image_path)

    payload = {
        "model": "MiniCPM-V-4.6",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    "Look very carefully at this image and provide a thorough, specific description. "
                    "Focus on: exact objects and their positions, any readable text or numbers, "
                    "dominant colors and lighting, estimated setting or environment, "
                    "and anything unusual or noteworthy. Be as precise and detailed as possible."
                )},
            ],
        }],
        "max_tokens": 700,  # higher budget for the second pass
    }

    return call_api(VISION_URL, payload)


# ─────────────────────────────────────────────
# Stage 2 — Planner: Decide next action
# ─────────────────────────────────────────────

def plan_next_step(description: str, memory: list[dict]) -> str:
    """MiniCPM4.1-8B reads the vision output AND recent run history to decide next action.

    Returns one of:
      'generate_report'  — description is clear, proceed to report writing
      'reinspect'        — description is vague or missing key details
      'flag_unclear'     — image seems unrelated, corrupted, or confusing
    """
    print("[Stage 2] Planner evaluating vision output...")

    # Format memory as a readable summary for the prompt
    if memory:
        memory_lines = "\n".join(
            f"  - {r['ts'][:16]} | image: {r['image']} | decision: {r['decision']} "
            f"| confidence: {r['confidence']}/10 | tools: {r['tools']}"
            for r in memory
        )
        memory_block = f"""Recent run history (use this to spot patterns, e.g. repeated blurry images):
{memory_lines}

"""
    else:
        memory_block = "No previous runs recorded yet.\n\n"

    prompt = f"""You are a planning agent in an image analysis pipeline.

{memory_block}A vision model just described a new image as follows:
{description}

Your job is to decide the best next action based on the quality and clarity of that description.
You may also consider patterns from the run history above (e.g. if recent images from the same
source have been consistently vague or unclear, weight that in your decision).

Rules:
- Choose 'generate_report' if the description is specific, detailed, and makes sense.
- Choose 'reinspect' if the description is vague, very short, or seems to be missing important details.
- Choose 'flag_unclear' if the image appears corrupted, completely unrelated to any real-world subject, or the description is incoherent.

Reply with ONLY one of these exact strings (no punctuation, no explanation):
generate_report
reinspect
flag_unclear

Your decision:"""

    payload = {
        "model": "MiniCPM4.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,  # must be high enough for <think>...</think> to close before the answer
    }

    raw = call_api(LANGUAGE_URL, payload).strip()

    # Strip complete <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Also strip any unclosed <think> block (model cut off mid-reasoning)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

    raw = raw.lower()
    # Normalise — guard against the model adding punctuation or extra words
    for valid in ("generate_report", "reinspect", "flag_unclear"):
        if valid in raw:
            return valid

    # Default to reinspect if the model returns something unexpected
    print(f"  [Planner] Unexpected response '{raw}' — defaulting to 'reinspect'")
    return "reinspect"


# ─────────────────────────────────────────────
# Stage 3 — Brain: Report generation
# ─────────────────────────────────────────────

def generate_report(description: str, image_path: str) -> str:
    print("[Stage 3] Generating structured report...")

    prompt = f"""You are a professional analyst writing a structured image report.

Image file: {image_path}
Analyzed at: {datetime.now().strftime("%Y-%m-%d %H:%M")}

Based on the following image description, produce a report with exactly these sections:

1. Summary
2. Key Observations
3. Potential Use Cases or Insights

Image Description:
{description}
"""

    payload = {
        "model": "MiniCPM4.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
    }

    return call_api(LANGUAGE_URL, payload)


# ─────────────────────────────────────────────
# Stage 4 — Critic: Review report quality
# ─────────────────────────────────────────────

def review_report(report: str) -> tuple[str, int, str]:
    """MiniCPM4.1-8B reads the finished report, judges quality, and assigns a confidence score.

    Returns a tuple of:
      (verdict, confidence, feedback)
      verdict    → 'approved' or 'regenerate'
      confidence → integer 1–10 (how certain the agent is the report is accurate/complete)
      feedback   → one sentence explaining what's missing or confirming quality
    """
    print("[Stage 4] Critic reviewing report quality...")

    prompt = f"""You are a quality-control critic for image analysis reports.

Read the following report and evaluate it:

--- REPORT START ---
{report}
--- REPORT END ---

A good report must:
1. Have all three sections: Summary, Key Observations, Potential Use Cases or Insights
2. Each section must contain at least two sentences of real content (not placeholders)
3. The content must be coherent and specific — not vague filler

Reply in exactly this format (3 lines, nothing else):
Line 1: approved   OR   regenerate
Line 2: confidence score as a single integer from 1 (very uncertain) to 10 (very confident)
Line 3: one sentence explaining your verdict

Your review:"""

    payload = {
        "model": "MiniCPM4.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,  # enough for <think> block + three-line answer
    }

    raw = call_api(LANGUAGE_URL, payload).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    verdict_line    = lines[0].lower() if len(lines) > 0 else ""
    confidence_line = lines[1]         if len(lines) > 1 else "5"
    feedback        = lines[2]         if len(lines) > 2 else "No specific feedback provided."

    # Parse verdict
    if "approved" in verdict_line:
        verdict = "approved"
    elif "regenerate" in verdict_line:
        verdict = "regenerate"
    else:
        print(f"  [Critic] Unexpected verdict '{verdict_line}' — defaulting to 'approved'")
        verdict = "approved"

    # Parse confidence — extract first integer found
    confidence = 5  # safe default
    match = re.search(r"\d+", confidence_line)
    if match:
        confidence = max(1, min(10, int(match.group())))

    print(f"  [Critic] Verdict: {verdict} | Confidence: {confidence}/10")
    return verdict, confidence, feedback


# ─────────────────────────────────────────────
# Tools — callable actions with dynamic arguments
# ─────────────────────────────────────────────

def tool_save_to_db(report: str, image_path: str, confidence: int) -> None:
    """Appends the report + confidence score as a JSON record to reports_db.jsonl."""
    db_path = Path("reports_db.jsonl")
    record = {
        "timestamp":  datetime.now().isoformat(),
        "image":      image_path,
        "confidence": confidence,
        "report":     report,
    }
    with db_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  [Tool: save_to_db] Record (confidence {confidence}/10) appended to {db_path}")


def tool_send_alert(image_path: str, message: str) -> None:
    """Sends an alert using the agent-written message.
    In production this would call an email/Slack/SMS API."""
    print(f"  [Tool: send_alert] *** ALERT SENT ***")
    print(f"    Image  : {image_path}")
    print(f"    Message: {message}")
    # e.g. requests.post(SLACK_WEBHOOK, json={"text": message})


def tool_crop_and_reanalyze(image_path: str, region_hint: str) -> str:
    """Re-analyzes a specific region identified by the agent.
    In production: PIL crop using region_hint, then call extract_image_info().
    Returns a note appended to the report."""
    print(f"  [Tool: crop_and_reanalyze] Region of interest: '{region_hint}'")
    # e.g. from PIL import Image; img = Image.open(image_path); crop = img.crop(parse(region_hint))
    return f"(Note: Region '{region_hint}' was identified for closer inspection. Crop-and-reanalyze would run here in production.)"


def tool_flag_for_human_review(image_path: str, confidence: int, reason: str) -> None:
    """Flags a low-confidence report for human review.
    In production this would write to a review queue or send a notification."""
    print(f"  [Tool: flag_for_human_review] *** FLAGGED ***")
    print(f"    Image     : {image_path}")
    print(f"    Confidence: {confidence}/10 (below threshold of {CONFIDENCE_THRESHOLD})")
    print(f"    Reason    : {reason}")
    flag_path = Path("human_review_queue.jsonl")
    with flag_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(),
            "image": image_path,
            "confidence": confidence,
            "reason": reason,
        }) + "\n")
    print(f"    Logged to : {flag_path}")


# ─────────────────────────────────────────────
# Stage 5 — Tool Picker: Choose tools + write arguments
# ─────────────────────────────────────────────

AVAILABLE_TOOLS = ["save_to_db", "send_alert", "crop_and_reanalyze"]

def pick_tools(report: str, confidence: int) -> list[dict]:
    """MiniCPM4.1-8B reads the report and decides which tools to call AND what to say.

    Returns a list of dicts: [{"tool": "send_alert", "args": {"message": "..."}}, ...]
    save_to_db is always included. flag_for_human_review is injected automatically
    if confidence is below CONFIDENCE_THRESHOLD — no model call needed for that one.
    """
    print("[Stage 5] Tool Picker selecting actions and writing arguments...")

    prompt = f"""You are an action-selection agent. You have read an image analysis report
and must decide which tools to run and provide the exact arguments for each.

--- REPORT ---
{report}
--- END REPORT ---

Available tools (save_to_db always runs — decide the others):
- send_alert        : call if the report mentions anything urgent, dangerous, critical, or risky.
                      You must write the alert message yourself — make it specific to the report.
- crop_and_reanalyze: call if a specific region or detail deserves a closer look.
                      You must name the region — e.g. "top-left corner", "person's face", "text label".

Reply with ONLY a JSON array. Each element has "tool" and "args". Example:
[
  {{"tool": "save_to_db", "args": {{}}}},
  {{"tool": "send_alert", "args": {{"message": "Damaged equipment detected in zone 3 — immediate inspection required."}}}},
  {{"tool": "crop_and_reanalyze", "args": {{"region_hint": "bottom-right area with faded text"}}}}
]

Always include save_to_db. Only include the others if genuinely warranted.

Your selection:"""

    payload = {
        "model": "MiniCPM4.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
    }

    raw = call_api(LANGUAGE_URL, payload).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"<think>.*",          "", raw, flags=re.DOTALL).strip()
    raw = raw.replace('\\"', '"')

    # Extract JSON array from response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            tool_calls = json.loads(match.group())
            # Validate tool names
            valid = [t for t in tool_calls if t.get("tool") in AVAILABLE_TOOLS + ["flag_for_human_review"]]
            # Ensure save_to_db is present
            names = [t["tool"] for t in valid]
            if "save_to_db" not in names:
                valid.insert(0, {"tool": "save_to_db", "args": {}})
            return valid
        except (json.JSONDecodeError, AttributeError):
            pass

    print(f"  [Tool Picker] Could not parse response — defaulting to save_to_db only")
    return [{"tool": "save_to_db", "args": {}}]


def run_tools(tool_calls: list[dict], report: str, image_path: str, confidence: int) -> str:
    """Executes each selected tool with its agent-written arguments."""
    # Inject flag_for_human_review automatically for low confidence — no model call needed
    names = [t["tool"] for t in tool_calls]
    if confidence < CONFIDENCE_THRESHOLD and "flag_for_human_review" not in names:
        tool_calls.append({
            "tool": "flag_for_human_review",
            "args": {"reason": f"Confidence score {confidence}/10 is below threshold {CONFIDENCE_THRESHOLD}."}
        })

    print(f"\n[Tools to run]: {[t['tool'] for t in tool_calls]}")

    for call in tool_calls:
        tool = call["tool"]
        args = call.get("args", {})

        if tool == "save_to_db":
            tool_save_to_db(report, image_path, confidence)
        elif tool == "send_alert":
            message = args.get("message", "Alert: image flagged by analysis pipeline.")
            tool_send_alert(image_path, message)
        elif tool == "crop_and_reanalyze":
            region_hint = args.get("region_hint", "unspecified region")
            note = tool_crop_and_reanalyze(image_path, region_hint)
            report += f"\n\n{note}"
        elif tool == "flag_for_human_review":
            reason = args.get("reason", "Low confidence score.")
            tool_flag_for_human_review(image_path, confidence, reason)

    return report


# ─────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────

def run_pipeline(image_path: str, output_path: str) -> None:
    confidence    = 5   # default; overwritten by Critic
    planner_decision = "generate_report"

    # Stage 0 — load memory so the Planner can learn from past runs
    memory = load_memory()
    print(f"[Stage 0] Memory loaded: {len(memory)} recent run(s)")

    # Stage 1 — always runs
    description = extract_image_info(image_path)
    print("\n--- Vision Output ---\n", description)

    # Stage 2 — Planner decides what happens next (with memory context)
    planner_decision = plan_next_step(description, memory)
    print(f"\n[Planner Decision]: {planner_decision}")

    if planner_decision == "reinspect":
        # Stage 1b — re-run vision with a sharper prompt
        print("\n[Re-inspecting with more specific prompt...]")
        description = extract_image_info_detailed(image_path)
        print("\n--- Detailed Vision Output ---\n", description)

    elif planner_decision == "flag_unclear":
        # Exit early — no report worth writing
        notice = (
            f"Image Analysis Notice\n"
            f"=====================\n"
            f"File: {image_path}\n"
            f"Analyzed at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"The image could not be analyzed clearly.\n"
            f"The vision model's description was incoherent or unrelated to a real-world subject.\n"
            f"Please check the image file and try again."
        )
        save_report(notice, output_path)
        print("[Pipeline] Exiting early — image flagged as unclear.")
        save_memory(image_path, planner_decision, confidence=0, verdict="skipped", tools_called=[])
        return

    # Stage 3 + 4 — generate report, Critic reviews + scores, loop if needed
    report = ""
    for attempt in range(1, MAX_RETRIES + 2):
        print(f"\n[Attempt {attempt}] Generating report...")
        report = generate_report(description, image_path)
        print("\n--- Report Draft ---\n", report)

        # Stage 4 — Critic reviews and scores
        verdict, confidence, feedback = review_report(report)
        print(f"[Critic Verdict]: {verdict} | Confidence: {confidence}/10 | {feedback}")

        if verdict == "approved":
            print("[Pipeline] Report approved.")
            break

        if attempt <= MAX_RETRIES:
            print(f"[Pipeline] Regenerating (attempt {attempt}/{MAX_RETRIES})...")
            description = (
                f"{description}\n\n"
                f"[Critic note from previous attempt: {feedback} "
                f"Please make sure all three sections are fully developed.]"
            )
        else:
            print("[Pipeline] Max retries reached — saving best available report.")

    # Save report
    save_report(report, output_path)

    # Stage 5 — Tool Picker: agent picks tools AND writes their arguments
    tool_calls = pick_tools(report, confidence)
    report = run_tools(tool_calls, report, image_path, confidence)

    # Re-save if crop_and_reanalyze appended a note
    if any(t["tool"] == "crop_and_reanalyze" for t in tool_calls):
        save_report(report, output_path)

    # Write this run to memory for future pipelines to learn from
    tools_called = [t["tool"] for t in tool_calls]
    save_memory(image_path, planner_decision, confidence, verdict, tools_called)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_report_pipeline.py <image_path> [output.txt]")
        sys.exit(1)

    image_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "report.txt"

    run_pipeline(image_path, output_path)
