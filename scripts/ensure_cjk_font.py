#!/usr/bin/env python3
"""Provision a CJK font for honest Linux Chromium screenshot acceptance.

The font is fetched only into the ephemeral CI runner; it is not checked in,
packed in Android, or distributed with SeetheTruth. Noto Sans CJK is SIL OFL
1.1: https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE
"""
from pathlib import Path
from subprocess import check_output, run
from urllib.request import urlopen

# Published Sans2.004 release tag, rather than an unfrozen main-branch path.
FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)
FONT_DIR = Path.home() / ".local/share/fonts/seethetruth-qa"


def installed():
    match = check_output(
        ["fc-match", "-f", "%{family}", "Noto Sans CJK SC:lang=zh-cn"],
        text=True,
    )
    return "Noto Sans CJK SC" in match


def main():
    if installed():
        print("SEETHETRUTH_CJK_FONT_READY: Noto Sans CJK SC already available")
        return
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    font = FONT_DIR / "NotoSansCJKsc-Regular.otf"
    with urlopen(FONT_URL, timeout=120) as response:
        # Bound the remote asset so an error response cannot silently pass.
        data = response.read(35_000_001)
    if len(data) < 1_000_000 or len(data) > 35_000_000 or data[:4] != b"OTTO":
        raise RuntimeError("Noto CJK download is missing or not a valid-sized OTF")
    font.write_bytes(data)
    run(["fc-cache", "-f", str(FONT_DIR)], check=True)
    if not installed():
        raise RuntimeError("Chromium QA is missing Simplified Chinese glyphs")
    print("SEETHETRUTH_CJK_FONT_READY: Noto Sans CJK SC provisioned for QA")


if __name__ == "__main__":
    main()
