import os
import json

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "output")
in_json = os.path.join(OUT_DIR, "live_trade_signals.json")

# Read JSON file
with open(in_json, "r") as f:
    data = json.load(f)

# Handle no signals case
if isinstance(data, dict) and "message" in data:
    print("No signals found")
    exit(0)

# If wrapped inside {"trades": [...]}
if isinstance(data, dict) and "trades" in data:
    trades = data["trades"]
else:
    trades = data if isinstance(data, list) else [data]

# Format trade info
lines = []
for t in trades:
    lines.append(
f"""🚀 Trade Alert

Stock: {t['Stock']}
Side: {t['Side']}
Entry: ₹ {t['Entry']}
Stop Loss: ₹ {t['StopLoss']}
Target: ₹ {t['Target']}
Shares: {t['Shares']}
Confidence: {t['Confidence']}
Timestamp: {t['Timestamp']}

⚠️ Automated signal — confirm liquidity & slippage before trading."""
    )

# Join multiple trades
output = "\n\n---\n\n".join(lines)
print(output)
