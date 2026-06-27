
import requests
import base64
import sys

# --- API Config ---
TOKEN = "sk-minicpm-V8bcD-YTAMxECagaKOnbwTCN69IlN2LhSezGOgq2Ues"
VISION_URL = "http://35.203.155.71:8003/v1/chat/completions"  # MiniCPM-V-4.6
LANGUAGE_URL = "http://35.203.155.71:8001/v1/chat/completions"  # MiniCPM4.1-8B

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


# --- Stage 1: Extract image info with the vision model ---
def extract_image_info(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "MiniCPM-V-4.6",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    "Describe this image in detail. "
                    "List all objects, text, people, colors, and any notable context you observe."
                )}
            ]
        }],
        "max_tokens": 500
    }

    resp = requests.post(VISION_URL, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --- Stage 2: Generate a structured report with the language model ---
def generate_report(image_description: str) -> str:
    prompt = f"""You are a professional report writer.
Based on the following image description, write a concise structured report with these sections:
1. Summary
2. Key Observations
3. Potential Use Cases or Insights

Image Description:
{image_description}
"""

    payload = {
        "model": "MiniCPM4.1-8B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600
    }

    resp = requests.post(LANGUAGE_URL, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --- Pipeline ---
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    print(f"[Stage 1] Sending {image_path} to MiniCPM-V-4.6 for vision extraction...\n")
    description = extract_image_info(image_path)
    print("Vision output:\n", description)

    print("\n[Stage 2] Sending description to MiniCPM4.1-8B for report generation...\n")
    report = generate_report(description)
    print("Final Report:\n", report)
