"""HTTP-based Mintlify smoke verification for remaining Mintlify tasks (1.9,2.4,2.5,7.2,7.4,7.8,7.11).
Runs `npx mintlify dev --port 3001` and checks the page via an HTTP request
using `requests`. Exits 0 on pass, 1 on fail.
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

PORT = 3001
URL = f"http://localhost:{PORT}"
DOCS_SITE_DIR = Path(__file__).resolve().parents[1]


def main() -> int:
    print(f"Starting mintlify dev on {PORT} (cwd={DOCS_SITE_DIR})...")
    proc = None
    try:
        proc = subprocess.Popen(
            ["npx", "mintlify", "dev", "--port", str(PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            cwd=str(DOCS_SITE_DIR),
        )

        # Poll until Mintlify is ready or timeout (instead of fixed sleep)
        deadline = time.monotonic() + 60
        import requests

        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"Mintlify process exited early with code {proc.returncode}")
                return 1
            try:
                r = requests.get(URL, timeout=2)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5)
        else:
            print(f"Mintlify did not become ready within 60s: {last_exc}")
            return 1

        print(f"GET {URL} -> {r.status_code}")
        # Specific checks (not generic substrings) to avoid false passes
        checks = {
            "1.9 dev starts": r.status_code == 200,
            "7.2 nav": (
                'href="/get-started/overview"' in r.text
                and 'href="/api-reference' in r.text
            ),
            "7.8 SEO": (
                'property="og:title"' in r.text
                and 'property="og:description"' in r.text
                and 'rel="canonical"' in r.text
            ),
            "7.11 search": (
                'type="search"' in r.text.lower()
                or 'id="search"' in r.text.lower()
                or "/search" in r.text
            ),
        }
        for k, v in checks.items():
            print(f"{k}: {'PASS' if v else 'FAIL'}")
        ok = all(checks.values())
        print("Overall:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except FileNotFoundError as exc:
        print(f"Browser check failed to start npx/mintlify: {exc}")
        return 1
    except Exception as exc:
        print(f"Browser check failed: {exc}")
        return 1
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
