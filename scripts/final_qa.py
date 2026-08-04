from pathlib import Path

path = Path("scripts/final_qa.py")

text = path.read_text(encoding="utf-8")

text = text.replace(
    'print(f"\n$ {\' \'.join(command)}")'.replace("\\n", "\n"),
    'print(f"\\\\n$ {\' \'.join(command)}")'
)

# اگر رشته واقعاً دو خطی شده باشد:
text = text.replace(
    'print(f"\n$ ',
    'print(f"\\\\n$ '
)

path.write_text(text, encoding="utf-8")

print("Fixed.")