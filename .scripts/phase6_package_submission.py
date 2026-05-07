"""
Phase 6: Package the assignment for submission.

Steps
-----
1. Make a copy of the executed notebook with the spec-compliant filename
   ``a2-code-530839244-540958494-550120560-550053316.ipynb`` (single
   ``.ipynb`` extension, no double).
2. Convert the notebook to HTML, then to PDF via headless Google Chrome.
3. Run a quick anonymisation check that warns if any group-member name or
   email lands in the notebook output.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

SOURCE_IPYNB = ROOT / "a2-code-530839244-540958494-550120560-550053316.ipynb"
SUBMIT_DIR   = ROOT / "Submission"
SUBMIT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Spec-compliant ipynb copy
# ------------------------------------------------------------------
shutil.copy2(SOURCE_IPYNB, SUBMIT_DIR / SOURCE_IPYNB.name)
print(f"Copied notebook -> {SOURCE_IPYNB.name} (Submission/)")

# ------------------------------------------------------------------
# 2. Notebook -> HTML
# ------------------------------------------------------------------
HTML_PATH = SUBMIT_DIR / (SOURCE_IPYNB.stem + ".html")
subprocess.run(
    [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "html",
        "--output-dir", str(SUBMIT_DIR),
        "--output", SOURCE_IPYNB.stem,
        str(SOURCE_IPYNB),
    ],
    check=True,
)
print(f"Wrote HTML -> {HTML_PATH}")

# ------------------------------------------------------------------
# 3. HTML -> PDF via headless Chrome
# ------------------------------------------------------------------
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_PATH = SUBMIT_DIR / (SOURCE_IPYNB.stem + ".pdf")
if Path(CHROME).exists():
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            f"file://{HTML_PATH.as_posix()}",
        ],
        check=True,
    )
    print(f"Wrote PDF  -> {PDF_PATH}")
else:
    print("Chrome not found — please convert the HTML to PDF manually "
          "(File > Print > Save as PDF in any browser).")

# ------------------------------------------------------------------
# 4. Quick anonymisation check
# ------------------------------------------------------------------
NAMES = ["Luca", "Minagawa", "Dylan", "George"]    # extend if you like
EMAILS = ["@uni.sydney.edu.au", "@gmail.com", "@sydney.edu.au"]
text = SOURCE_IPYNB.read_text(encoding="utf-8")
hits = [needle for needle in NAMES + EMAILS if needle in text]
if hits:
    print(f"WARNING — possibly identifying tokens still present in the "
          f"notebook: {hits!r}")
else:
    print("Anonymisation check OK — no listed names/emails found in the notebook.")

print()
print("Submission artefacts in:", SUBMIT_DIR)
for p in sorted(SUBMIT_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size//1024:>5} KB)")
