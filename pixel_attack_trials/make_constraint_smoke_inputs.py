"""Create synthetic engineering fixtures, not research attack examples."""
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parent / "constraint_smoke_inputs"
    root.mkdir(exist_ok=True)
    image = Image.new("RGB", (224, 224), (180, 180, 180))
    ImageDraw.Draw(image).ellipse((65, 65, 159, 159), fill=(40, 80, 220))
    image.save(root / "synthetic.png")
    flower = Image.new("RGBA", (56, 56), (0, 0, 0, 0))
    draw = ImageDraw.Draw(flower)
    for k in range(5):
        angle = 2*math.pi*k/5
        x, y = 28+13*math.cos(angle), 28+13*math.sin(angle)
        draw.ellipse((x-11, y-11, x+11, y+11), fill=(230, 100, 160, 255))
    draw.ellipse((20, 20, 36, 36), fill=(240, 200, 40, 255))
    flower.save(root / "synthetic_flower.png")
    with (root / "marker_questions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "category", "question", "normal_answer", "target_answer", "target_contains"])
        writer.writeheader()
        for split, question in [("train", "Describe the main object in the picture."),
                                ("test", "What shape and color is in the center?")]:
            writer.writerow(dict(split=split, category="marker", question=question,
                                 normal_answer="A blue circle.", target_answer="CONTROL_MARKER_42", target_contains="CONTROL_MARKER_42"))
    tasks = []
    for kind in ("border", "lowfreq", "flower", "sparse"):
        task = dict(image="synthetic.png", csv="marker_questions.csv", constraint=kind,
                    steps=1, max_train_rows=1, questions_per_step=1, max_visual_tokens=64,
                    eval_rows=1, patch_width=56, patch_height=56)
        if kind == "flower":
            task["flower_image"] = "synthetic_flower.png"
        tasks.append(task)
    (root / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(root / "tasks.json")


if __name__ == "__main__":
    main()
