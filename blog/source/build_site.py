"""Build the English blog and its supporting HTML notes."""
from pathlib import Path
from html import escape
import re
import struct
import markdown

ROOT = Path(__file__).resolve().parents[1]

FIGURES = {
    "assets/circuit_story_en.svg": ("figure-story", "From diagnosis to learned updates", 1200, 920),
    "assets/method_overview.png": ("figure-method", "Figure 1 · C-HyperNet workflow", None, None),
    "assets/hypernet_architecture.png": ("figure-architecture", "Figure 2 · HyperNet architecture", None, None),
    "assets/lean_tasks.png": ("figure-lean-tasks", "Figure 3 · Lean tasks", None, None),
    "assets/lean_eval_protocol.png": ("figure-lean-protocol", "Figure 4 · Compositional training and evaluation", None, None),
    "assets/hypernet_assisted_localization.png": ("figure-localization", "Figure 5 · HyperNet-assisted joint ranking", None, None),
}

SUPPORTING_NOTES = [
    ("probe_results_en.md", "probe_results.html", "analysis-1", "Section 4.1 · Compositional analysis"),
    ("composition_nodes_en.md", "composition_nodes.html", "analysis-1", "Section 4.2 · Compositional analysis"),
    ("localization_results_en.md", "localization_results.html", "analysis-2", "Section 5 · Functional nodes"),
    ("ttft_results_en.md", "ttft_results.html", "experiments", "Section 3 · Main experiments"),
]

NOTE_LINK_MAP = {md_name: html_name for md_name, html_name, _, _ in SUPPORTING_NOTES}

HEADER_HTML = '''  <header class="site-header">
    <a class="wordmark" href="index.html#top"><svg viewBox="0 0 32 32" width="29" height="29" aria-hidden="true"><path d="M7 8 16 5 25 13 16 25 7 20 7 8 16 15 25 13M16 5v20M7 20l18-7" fill="none" stroke="currentColor" stroke-width="1.4"/><g fill="currentColor"><circle cx="7" cy="8" r="3"/><circle cx="16" cy="5" r="3"/><circle cx="25" cy="13" r="3"/><circle cx="16" cy="25" r="3"/><circle cx="7" cy="20" r="3"/></g><circle cx="16" cy="15" r="3" fill="#cb9d86"/></svg><span>Circuit-HyperNet</span></a>
    <a class="header-link" href="__HEADER_LINK_HREF__">__HEADER_LINK_TEXT__</a>
  </header>'''


def wrap_tables_and_links(rendered):
    rendered = re.sub(
        r'<table>.*?</table>',
        lambda match: '<div class="result-table" role="region" aria-label="Results table; scroll horizontally if needed" tabindex="0">'
        + match.group(0).replace('<th ', '<th scope="col" ').replace('<th>', '<th scope="col">')
        + '</div>',
        rendered,
        flags=re.S,
    )
    for md_name, html_name in NOTE_LINK_MAP.items():
        rendered = rendered.replace(f'href="{md_name}"', f'href="{html_name}"')
    return re.sub(r'<a href="(https?://[^"]+)"', r'<a target="_blank" rel="noopener noreferrer" href="\1"', rendered)


def render_section(text, expected_figures):
    rendered = markdown.markdown(text, extensions=["extra", "sane_lists"])
    pattern = r'<p><img alt="([^"]+)" src="(assets/[^"]+)"\s*/></p>\s*((?:<p><em>.*?</em></p>\s*)+)'

    def render_figure(match):
        alt, path, caption = match.groups()
        caption = caption.strip()
        anchor, title, width, height = FIGURES[path]
        if width is None:
            width, height = struct.unpack(">II", (ROOT / path).read_bytes()[16:24])
        tall_class = " story-figure--tall" if height > width else ""
        return f'''<figure class="story-figure{tall_class}" id="{anchor}">
  <div class="figure-toolbar"><span>{escape(title)}</span><button class="figure-open" type="button" aria-haspopup="dialog">Expand figure <span aria-hidden="true">↗</span></button></div>
  <a class="figure-fallback" href="{path}" target="_blank" rel="noopener" aria-label="Open the figure at full size"><img src="{path}" width="{width}" height="{height}" alt="{alt}" decoding="async"></a>
  <figcaption>{caption}</figcaption>
</figure>'''

    rendered, count = re.subn(pattern, render_figure, rendered, flags=re.S)
    assert count == expected_figures, f"Expected {expected_figures} figures, found {count}"
    return wrap_tables_and_links(rendered)


def build_supporting_notes():
    for md_name, html_name, return_anchor, section_label in SUPPORTING_NOTES:
        raw = (ROOT / md_name).read_text(encoding="utf-8").strip()
        first_line, body = raw.split("\n", 1)
        assert first_line.startswith("# "), md_name
        title = first_line[2:].strip()
        body_html = wrap_tables_and_links(markdown.markdown(body.strip(), extensions=["extra", "sane_lists"]))
        header = HEADER_HTML.replace("__HEADER_LINK_HREF__", f"index.html#{return_anchor}").replace(
            "__HEADER_LINK_TEXT__", "Back to article <span aria-hidden=\"true\">←</span>"
        )
        page_html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{escape(title)} · Circuit-HyperNet</title>
  <link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="styles.css?v=13">
</head>
<body>
  <a class="skip-link" href="#note-content">Skip to note</a>
{header}
  <main class="note-main" id="note-content">
    <div class="note-header">
      <a class="note-back" href="index.html#{return_anchor}"><span aria-hidden="true">←</span> Back to {escape(section_label)}</a>
      <h1>{escape(title)}</h1>
      <div class="note-meta"><span>Supporting record for Circuit-HyperNet</span><span class="meta-divider" aria-hidden="true">·</span><a href="{md_name}" download>Download Markdown <span aria-hidden="true">↓</span></a></div>
    </div>
    <article class="note-body article" aria-label="{escape(title)}">
{body_html}
    </article>
  </main>
  <footer class="site-footer"><span>Circuit-HyperNet</span><span>Supporting experimental record</span><a href="index.html#{return_anchor}">Back to article ←</a></footer>
</body>
</html>
'''
        (ROOT / html_name).write_text(page_html, encoding="utf-8", newline="\n")
        print("Built", ROOT / html_name)


SECTION_FILES = [
    ("abstract", "", "abstract_en.md", "Abstract", 0),
    ("motivation", "01", "motivation_en.md", "Motivation", 1),
    ("method", "02", "method_en.md", "Method", 2),
    ("experiments", "03", "experiments_en.md", "Main experiments", 2),
    ("analysis-1", "04", "analysis1_en.md", "Compositional analysis", 0),
    ("analysis-2", "05", "localization_en.md", "Functional nodes", 1),
    ("related-work", "06", "related_work_en.md", "Related work", 0),
    ("discussion", "07", "discussion_en.md", "Discussion", 0),
    ("references", "", "references_en.md", "References", 0),
    ("appendix", "", "appendix_en.md", "Appendix", 0),
]

article_title = "Circuit-HyperNet: Guiding parameter generation and compositional reasoning with internal computation"
article_sections = []
section_content = {}
markdown_sections = []
for anchor, number, filename, short, figure_count in SECTION_FILES:
    section = (ROOT / filename).read_text(encoding="utf-8").strip()
    heading_line, body = section.split("\n", 1)
    assert heading_line.startswith("## "), filename
    heading = re.sub(r"^[1-7]\. ", "", heading_line[3:])
    article_sections.append((anchor, number, heading, short))
    section_content[anchor] = render_section(body.strip(), figure_count)
    markdown_sections.append(section)

(ROOT / "article_en.md").write_text(
    "# " + article_title + "\n\n" + "\n\n---\n\n".join(markdown_sections) + "\n",
    encoding="utf-8",
    newline="\n",
)

nav = "\n".join(
    f'<a href="#{anchor}"><span class="nav-number">{number or "—"}</span><span>{escape(short)}</span></a>'
    for anchor, number, _, short in article_sections
)
article = []
for anchor, number, heading, _ in article_sections:
    number_html = f'<span class="section-number">{number}</span> ' if number else ""
    heading_html = f'<div class="section-heading"><h2>{number_html}{escape(heading)}</h2></div>'
    section_body = section_content[anchor]
    article.append(
        f'<section id="{anchor}" class="article-section filled-section" aria-label="{escape(heading)}">{heading_html}{section_body}</section>'
    )
article_html = "\n".join(article)

main_header = HEADER_HTML.replace('href="index.html#top"', 'href="#top"').replace(
    "__HEADER_LINK_HREF__", "#abstract"
).replace("__HEADER_LINK_TEXT__", 'Read the article <span aria-hidden="true">↗</span>')

html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Circuit-HyperNet (C-HyperNet) learns per-query LoRA updates from internal computation, with experiments on Lean compositional reasoning, ARC-AGI-1, and functional-node localization.">
  <meta property="og:title" content="Circuit-HyperNet — Parameter Generation and Compositional Reasoning">
  <meta property="og:description" content="Learning per-query parameter updates from the computation behind a model's first answer.">
  <meta property="og:type" content="article">
  <title>Circuit-HyperNet · Research Blog</title>
  <link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="styles.css?v=14">
  <script src="main.js?v=4" defer></script>
</head>
<body>
  <a class="skip-link" href="#abstract">Skip to article</a>
__HEADER__
  <main id="top">
    <div class="hero">
      <p class="eyebrow"><span class="accent-dot"></span> CIRCUIT-HYPERNET <span class="eyebrow-divider">/</span> RESEARCH BLOG</p>
      <h1>__ARTICLE_TITLE__</h1>
      <p class="hero-deck">A small hypernetwork turns the computation behind a model’s first answer into a parameter update for its next attempt.</p>
      <div class="hero-meta"><span>Research Note</span><span class="meta-divider" aria-hidden="true">·</span><span>November 22, 2026</span></div>
    </div>
    <div class="reading-layout">
      <aside class="contents" aria-label="On this page"><p class="contents-label">IN THIS ARTICLE</p>__NAV__<a class="download-link" href="article_en.md" download>Full article as Markdown <span aria-hidden="true">↓</span></a></aside>
      <details class="mobile-contents"><summary>Contents</summary><nav aria-label="Article sections">__NAV__</nav></details>
      <article class="article" aria-label="Circuit-HyperNet research article">__CONTENT__</article>
    </div>
  </main>
  <footer class="site-footer"><span>Circuit-HyperNet</span><span>Internal computation → per-query parameter updates</span><a href="#top">Back to top ↑</a></footer>
  <dialog id="figure-dialog" aria-labelledby="figure-dialog-title">
    <div class="dialog-header"><h2 id="figure-dialog-title">From diagnosis to learned updates</h2><div class="dialog-actions"><button id="figure-zoom" type="button" aria-pressed="false">Actual size</button><button id="figure-close" type="button" autofocus>Close <span aria-hidden="true">×</span></button></div></div>
    <div class="figure-scroll"><img src="assets/circuit_story_en.svg" width="1200" height="920" alt="Three scenarios connecting internal computation, model errors, and parameter updates"></div>
  </dialog>
</body>
</html>
'''.replace("__ARTICLE_TITLE__", escape(article_title)).replace("__HEADER__", main_header).replace("__NAV__", nav).replace("__CONTENT__", article_html)

(ROOT / "index.html").write_text(html, encoding="utf-8", newline="\n")
print("Built", ROOT / "index.html")
build_supporting_notes()
