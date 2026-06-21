import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v14_session_management.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

############################ Session Configuration ###########################

# A session_id ties multiple arun() calls to the same underlying browser
# tab/context. Any state written to cookies, localStorage, or sessionStorage
# during one call is automatically available to the next call that shares the
# same session_id — just as if a real user were navigating within the same tab.
SESSION_ID = "v14_session_demo"

# Target URL used for both steps. Using example.com keeps the demo self-contained
# and avoids any external side-effects.
URL = "https://example.com/"

# JavaScript injected during Step 1: writes a value to sessionStorage so it can
# be verified in Step 2, and sets a data attribute on <body> to signal that the
# setup script has finished executing (used as a wait_for selector below).
SET_STATE_JS = """
sessionStorage.setItem("tutorial_state", "session-kept");
document.body.setAttribute("data-session-step", "1");
"""

# JavaScript injected during Step 2: reads back the sessionStorage value set in
# Step 1 to confirm it persisted across crawl calls. The value is emitted via
# console.log (captured by capture_console_messages) and also injected into the
# DOM as a <p> element so it can be used as a wait_for selector.
READ_STATE_JS = """
const value = sessionStorage.getItem("tutorial_state") || "missing";
console.log("tutorial_state=" + value);
document.body.insertAdjacentHTML("beforeend", `<p class="session-value">${value}</p>`);
"""

############################## Main Crawl Routine ############################


async def main() -> None:
    """Demonstrate Crawl4AI session management across two sequential arun() calls.

    Step 1 writes state into sessionStorage and marks the page with a data
    attribute.  Step 2 reuses the same browser tab (via session_id) to read
    that state back, proving it was preserved.  The session is then explicitly
    closed to free the underlying browser resource.
    """
    async with AsyncWebCrawler() as crawler:
        # --- Step 1: write state ---
        # Passing session_id tells Crawl4AI to open (or reuse) a named browser
        # tab for this request. SET_STATE_JS runs after the page loads and writes
        # to sessionStorage. wait_for blocks until the data attribute set by the
        # script is present, confirming the JS finished before we move on.
        step1 = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(
                session_id=SESSION_ID,
                cache_mode=CacheMode.BYPASS,
                js_code=SET_STATE_JS,
                wait_for="body[data-session-step='1']",
                verbose=False,
            ),
        )

        # --- Step 2: verify state ---
        # The same session_id routes this request to the exact same browser tab,
        # so the sessionStorage written in Step 1 is still present. READ_STATE_JS
        # retrieves the stored value and appends it to the DOM.
        # capture_console_messages=True tells Crawl4AI to intercept browser
        # console output and surface it in result.console_messages, letting us
        # inspect the console.log call made inside READ_STATE_JS.
        step2 = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(
                session_id=SESSION_ID,
                cache_mode=CacheMode.BYPASS,
                js_code=READ_STATE_JS,
                wait_for=".session-value",
                capture_console_messages=True,
                verbose=False,
            ),
        )

        # Explicitly close the named browser tab to release its memory and
        # process handle. Without this call the tab would linger until the
        # AsyncWebCrawler context manager exits; calling kill_session() earlier
        # is good practice when you know the session is no longer needed.
        await crawler.crawler_strategy.kill_session(SESSION_ID)

    # Report the success status of each step and dump the captured console
    # output so we can confirm the sessionStorage value was "session-kept"
    # rather than the fallback "missing".
    print("step1:", step1.success)
    print("step2:", step2.success)
    print("console:", getattr(step2, "console_messages", None))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
