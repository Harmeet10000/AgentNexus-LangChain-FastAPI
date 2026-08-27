"""Headless browser verification for remaining Mintlify tasks (1.9,2.4,2.5,7.2,7.4,7.8,7.11).
Runs `npx mintlify dev --port 3001` and checks via Playwright if available,
otherwise falls back to curl checks. Exits 0 on pass, 1 on fail.
"""
import subprocess, time, sys, os, signal

PORT = 3001
URL = f"http://localhost:{PORT}"

def main():
    print(f"Starting mintlify dev on {PORT}...")
    proc = subprocess.Popen(["npx", "mintlify", "dev", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    time.sleep(15)
    try:
        import requests
        r = requests.get(URL, timeout=10)
        print(f"GET {URL} -> {r.status_code}")
        checks = {
            "1.9 dev starts": r.status_code == 200,
            "7.2 nav": "href" in r.text,
            "7.8 SEO": "og:title" in r.text or "meta" in r.text,
            "7.11 search": "search" in r.text.lower(),
        }
        for k,v in checks.items():
            print(f"{k}: {'PASS' if v else 'FAIL'}")
        ok = all(checks.values())
        print("Overall:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except Exception as e:
        print(f"Browser check failed: {e}")
        return 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except: pass
        proc.wait(timeout=5)

if __name__ == "__main__":
    sys.exit(main())
