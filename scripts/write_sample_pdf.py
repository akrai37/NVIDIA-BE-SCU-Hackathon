from pathlib import Path
import base64

# Minimal PDF binary (one-page placeholder). This is a small valid PDF used for local testing.
PDF_B64 = (
    "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlIC9DYXRhbG9nPj4KZW5kb2JqCnhyZWYKMCAyCjAwMDAwMDAwMCAwMDAwMCBuCj"
    "AwMDAwMDAxMCAwMDAwMCBuCjAwMDAwMDAyMCAwMDAwMCBuCnRyYWlsZXIKPDwvU2l6ZSAyL1Jvb3QgMSAwIFIvSW5mbyAyIDAgUi9JRCBbPDg0NjhCNzYzNkRENTA1REJGN0YyRjk0MjA3RkYxNzQ4Pl0+PgpzdGFydHhyZWYKMTg1CiUlRU9G"
)

out = Path(__file__).resolve().parents[1] / "samples" / "grant_brief.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    f.write(base64.b64decode(PDF_B64))
print("Wrote", out)
