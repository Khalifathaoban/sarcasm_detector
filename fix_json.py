import json

# Read the current JSON array file
with open("Sarcasm_Headlines_Dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Write as line-delimited JSON
with open("Sarcasm_Headlines_Dataset_v2_fixed.json", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print("✅ Fixed JSON saved as: Sarcasm_Headlines_Dataset_v2_fixed.json")
