"""Front-end checks: the page has every control the demo needs and the JS parses."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"


def test_index_has_all_sections_and_controls():
    html = (WEB / "index.html").read_text()
    for tab in ("ask", "compare", "eval", "repos"):
        assert f'data-tab="{tab}"' in html and f'id="tab-{tab}"' in html
    for ctrl in ("modeSeg", "chunkSeg", "retrieverSel", "repoSelect", "compareQuestions", "addRepoForm",
                 "viewerGithub", "viewerClose"):
        assert f'id="{ctrl}"' in html
    assert 'name="viewport"' in html and 'lang="en"' in html


def test_every_id_used_in_js_exists_in_html():
    html = (WEB / "index.html").read_text()
    js = (WEB / "app.js").read_text()
    ids = set(re.findall(r'\$\("#([A-Za-z]+)"\)', js))
    dynamic = {"recallChart", "chartK", "kSeg", "qtable"}      # created by app.js itself
    missing = [i for i in ids - dynamic if f'id="{i}"' not in html]
    assert not missing, missing


def test_css_supports_dark_mode_and_reduced_motion():
    css = (WEB / "styles.css").read_text()
    assert "prefers-color-scheme: dark" in css and "prefers-reduced-motion" in css
    assert "[hidden] { display: none !important; }" in css


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_is_valid_javascript():
    out = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
