from pathlib import Path

VALID_CLASSES = set(range(6))  # 0–5

for split in ["images", "val"]:
    label_dir = Path("dataset/val") / split
    for label_file in label_dir.glob("*.txt"):
        new_lines = []
        for line in label_file.read_text().splitlines():
            cls, *rest = line.split()
            if int(cls) in VALID_CLASSES:
                new_lines.append(line)
        label_file.write_text("\n".join(new_lines))