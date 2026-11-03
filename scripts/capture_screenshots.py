"""
Drives the real running app to produce the README screenshots and demo GIF.

    pip install playwright pillow
    streamlit run src/app.py
    python scripts/capture_screenshots.py

Uses the installed Chrome, and spends a few cents of API credit on real answers.
"""
import io
import os
import time

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGES_DIR = os.path.join(ROOT_DIR, "docs", "images")
SAMPLE_DOC = os.path.join(ROOT_DIR, "examples", "zorblax.md")
APP_URL = "http://localhost:8501"
VIEWPORT = {"width": 1360, "height": 860}
TIMEOUT_MS = 180_000
QUESTION = "How often do moonkeys rotate, and what status code does an expired key return?"


class Recorder:
    """Collects frames for the GIF while the app is being used."""

    def __init__(self, page):
        self.page = page
        self.frames = []

    def frame(self, hold: int = 1):
        image = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        image = image.resize((960, int(960 * image.height / image.width)), Image.LANCZOS)
        self.frames.extend([image] * hold)

    def save(self, path: str):
        palette = self.frames[len(self.frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self.frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0, optimize=True)


def settle(page, seconds: float = 1.5):
    page.wait_for_function(
        "!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=TIMEOUT_MS
    )
    time.sleep(seconds)


def shot(page, name: str, full_page: bool = False):
    page.screenshot(path=os.path.join(IMAGES_DIR, f"{name}.png"), full_page=full_page)
    print(f"Saved {name}.png")


def scroll_main_to_bottom(page):
    page.evaluate("""
        [...document.querySelectorAll('*')]
            .filter(e => e.scrollHeight > e.clientHeight + 20 && getComputedStyle(e).overflowY === 'auto')
            .forEach(e => e.scrollTo(0, e.scrollHeight))
    """)
    time.sleep(0.5)


def ask(page, question: str, recorder: Recorder | None = None):
    box = page.get_by_test_id("stChatInputTextArea")
    box.click()
    box.type(question, delay=12)
    if recorder:
        recorder.frame(2)
    box.press("Enter")
    # The search can finish too quickly to catch its spinner, so don't wait for that.
    time.sleep(0.6)
    if recorder:
        recorder.frame(2)
    page.get_by_text("Sources", exact=True).last.wait_for(timeout=TIMEOUT_MS)
    settle(page)
    scroll_main_to_bottom(page)
    if recorder:
        recorder.frame(6)


def open_tab(page, label: str):
    page.get_by_role("link", name=label).first.click()
    settle(page)


def capture(page, scheme: str):
    page.emulate_media(color_scheme=scheme)
    suffix = "" if scheme == "light" else "-dark"

    # Upload flow, with a framework that exists nowhere else
    page.goto(APP_URL)
    page.get_by_text("Build knowledge base").wait_for(timeout=TIMEOUT_MS)
    settle(page)
    shot(page, f"upload{suffix}")
    if scheme != "light":
        return

    recorder = Recorder(page)
    recorder.frame(4)
    page.locator('[data-testid="stFileUploader"] input[type=file]').set_input_files(SAMPLE_DOC)
    page.get_by_text("zorblax.md").first.wait_for(timeout=TIMEOUT_MS)
    settle(page, 0.5)
    recorder.frame(3)

    page.get_by_role("button", name="Build knowledge base").click()
    page.get_by_text("Building your knowledge base").wait_for(timeout=TIMEOUT_MS)
    recorder.frame(2)
    time.sleep(2)
    recorder.frame(2)
    page.get_by_role("heading", name="Ask about zorblax.md").wait_for(timeout=TIMEOUT_MS)
    settle(page)
    recorder.frame(3)
    shot(page, "ready")

    ask(page, QUESTION, recorder)
    shot(page, "answer")

    box = page.get_by_test_id("stChatInputTextArea")
    box.click()
    box.type("How do I define a model in Django?", delay=12)
    recorder.frame(2)
    box.press("Enter")
    time.sleep(2)
    page.get_by_text("I couldn't find this in the documents").wait_for(timeout=TIMEOUT_MS)
    settle(page)
    scroll_main_to_bottom(page)
    recorder.frame(6)
    shot(page, "refusal")
    recorder.save(os.path.join(IMAGES_DIR, "demo.gif"))
    print("Saved demo.gif")

    # A source opened up, on the sample docs, where citations link to real pages
    page.get_by_role("button", name="Change documents").click()
    page.get_by_role("button", name="Try it with the FastAPI docs").click()
    page.get_by_role("heading", name="Ask about FastAPI docs").wait_for(timeout=TIMEOUT_MS)
    settle(page)
    ask(page, "How do I add custom headers to a response?")
    source = page.locator('[data-testid="stChatMessage"] [data-testid="stExpander"] summary').first
    source.click()
    time.sleep(1)
    source.evaluate("element => element.scrollIntoView({block: 'start'})")
    time.sleep(0.5)
    shot(page, "sources", full_page=False)

    open_tab(page, "Compare")
    page.get_by_label("Question").fill("How do I return a 404 error when an item does not exist?")
    page.get_by_role("button", name="Compare").click()
    page.get_by_role("tab", name="Baseline").wait_for(timeout=TIMEOUT_MS)
    settle(page)
    shot(page, "compare")

    open_tab(page, "Evaluation")
    time.sleep(2)
    shot(page, "evaluation")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for scheme in ("dark", "light"):
            context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1.5)
            page = context.new_page()
            page.set_default_timeout(TIMEOUT_MS)
            capture(page, scheme)
            context.close()
        browser.close()


if __name__ == "__main__":
    main()
