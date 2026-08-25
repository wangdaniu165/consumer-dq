"""Render ``docs/model.md`` to a standalone ``model.html`` (math via MathJax).

Run:  python -m src.export_model_html
Output: ``model.html`` at the project root — open it in any browser.

The markdown is converted to HTML with Python-Markdown; the LaTeX in the doc is
left untouched and rendered client-side by MathJax (single ``$`` for inline math,
``$$`` for display math).
"""

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "model.md"
OUT = ROOT / "model.html"

EXTS = ["tables", "fenced_code", "sane_lists"]


def render() -> str:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=EXTS)
    return TEMPLATE.replace("<!--BODY-->", body)


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Documentation &mdash; Consumer Delinquency vs Unemployment</title>
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    },
    options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'code'] }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 20px 80px;
    background: #fcfcfb; color: #0b0b0b;
    font: 16px/1.65 system-ui, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 820px; margin: 0 auto; }
  .back { font-size: 14px; margin: 0 0 24px; }
  .back a { color: #2a78d6; text-decoration: none; }
  h1 { font-size: 26px; line-height: 1.3; margin: 0 0 4px; }
  h2 { font-size: 21px; margin: 40px 0 12px; padding-top: 16px; border-top: 1px solid #e1e0d9; }
  h3 { font-size: 17px; margin: 28px 0 8px; }
  p, li { color: #1c1b18; }
  code {
    background: #efeeea; border-radius: 4px; padding: 1px 5px;
    font: 0.9em "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }
  pre {
    background: #17171a; color: #e6e6e4; border-radius: 8px;
    padding: 14px 16px; overflow-x: auto;
  }
  pre code { background: none; padding: 0; color: inherit; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }
  th, td { border: 1px solid #e1e0d9; padding: 7px 10px; text-align: left; }
  th { background: #f4f3ef; font-weight: 600; }
  blockquote { margin: 16px 0; padding: 2px 16px; border-left: 3px solid #c3c2b7; color: #56554f; }
  hr { border: none; border-top: 1px solid #e1e0d9; margin: 32px 0; }
  a { color: #2a78d6; }
  .mjx-math { font-size: 1.05em; }
</style>
</head>
<body>
<div class="wrap">
<p class="back"><a href="dashboard.html">&larr; Back to dashboard</a></p>
<!--BODY-->
</div>
</body>
</html>
"""


def main() -> None:
    html = render()
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
