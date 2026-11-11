"""
One-command launcher for the demo on Windows, macOS and Linux.

    python run_demo.py              # set up (first time only) and start the demo
    python run_demo.py --full       # also install the real-image detection tab
    python run_demo.py --check      # set up and run a self-check, then exit
    python run_demo.py --recreate   # delete and rebuild the private environment

What it does:
1. checks the Python version (3.10 - 3.13 recommended),
2. creates a private virtual environment in .venv/ (first run only),
3. installs requirements.txt into it (again only when requirements change),
4. starts the Gradio app and opens http://127.0.0.1:7860 in your browser.

Any extra options (--port 8000, --share, --no-browser) are passed to the app.
Uses only the Python standard library, so it runs before anything is
installed.
"""

import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
MARKER = os.path.join(VENV, ".demo-deps")
MIN_PY = (3, 10)
MAX_TESTED = (3, 13)
IS_WIN = os.name == "nt"


def say(msg=""):
    print(msg, flush=True)


def fail(msg, code=1):
    say("")
    say("ERROR: " + msg)
    sys.exit(code)


def venv_python():
    if IS_WIN:
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def run(cmd, **kw):
    return subprocess.call(cmd, cwd=ROOT, **kw)


def check_python():
    v = sys.version_info[:2]
    say(f"Python {sys.version.split()[0]} at {sys.executable}")
    if v < MIN_PY:
        fail("Python 3.10 or newer is needed (3.12 recommended).\n"
             "  Windows/macOS: install it from https://www.python.org/downloads/\n"
             "                 (Windows: tick 'Add python.exe to PATH').\n"
             "  macOS (Homebrew): brew install python@3.12\n"
             "  Ubuntu/Debian:    sudo apt install python3.12 python3.12-venv")
    if v > MAX_TESTED:
        say(f"WARNING: Python {v[0]}.{v[1]} is newer than the versions this "
            "demo was tested with (3.10 - 3.13).\n"
            "         If installing packages fails, install Python 3.12 and "
            "run again.")


def requirement_files(full):
    files = ["requirements.txt"]
    if full:
        files.append("requirements-perception.txt")
    return files


def deps_fingerprint(files):
    h = hashlib.sha256()
    h.update(sys.version.encode())
    for f in files:
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()


def create_venv():
    say("")
    say("Creating a private Python environment in .venv/ (first run only)...")
    rc = run([sys.executable, "-m", "venv", VENV])
    if rc != 0 or not os.path.exists(venv_python()):
        shutil.rmtree(VENV, ignore_errors=True)
        hint = ""
        if sys.platform.startswith("linux"):
            v = f"{sys.version_info[0]}.{sys.version_info[1]}"
            hint = ("\n  On Ubuntu/Debian install the venv module first:\n"
                    f"      sudo apt install python{v}-venv\n"
                    "  then run this command again.")
        fail("Could not create the virtual environment." + hint)


def venv_works():
    py = venv_python()
    if not os.path.exists(py):
        return False
    try:
        out = subprocess.run([py, "-c", "import sys; print(sys.version)"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.split()[:1] == \
        sys.version.split()[:1]


def install(files):
    py = venv_python()
    say("")
    say("Installing the demo's packages (first run only, needs internet, "
        "a few minutes)...")
    run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q",
         "--upgrade", "pip"])
    cmd = [py, "-m", "pip", "install", "--disable-pip-version-check",
           "--retries", "5", "--timeout", "60"]
    for f in files:
        cmd += ["-r", f]
    rc = run(cmd)
    if rc != 0:
        fail("Installing packages failed (see the messages above).\n"
             "  - Check your internet connection and run the command again.\n"
             "  - If you use a very new Python, install Python 3.12 and run "
             "again with --recreate.")


def main():
    args = sys.argv[1:]
    full = "--full" in args
    check = "--check" in args
    recreate = "--recreate" in args
    app_args = [a for a in args if a not in ("--full", "--check",
                                            "--recreate")]
    if any(a in ("-h", "--help") for a in app_args):
        say(__doc__)
        return 0

    os.chdir(ROOT)
    say("=== Warehouse robot obstacle-avoidance demo ===")
    check_python()

    if recreate and os.path.isdir(VENV):
        say("Removing the old .venv/ ...")
        shutil.rmtree(VENV, ignore_errors=True)
    if os.path.isdir(VENV) and not venv_works():
        say("The existing .venv/ was made by another Python or computer; "
            "rebuilding it.")
        shutil.rmtree(VENV, ignore_errors=True)
    if not os.path.isdir(VENV):
        create_venv()

    files = requirement_files(full)
    fp = deps_fingerprint(files)
    old = open(MARKER).read().strip() if os.path.exists(MARKER) else ""
    if old != fp:
        install(files)
        with open(MARKER, "w") as fh:
            fh.write(fp)
    else:
        say("Packages already installed.")

    env = dict(os.environ)
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    py = venv_python()

    if check:
        say("")
        say("Running the self-check...")
        return run([py, "-m", "src.app.selfcheck"], env=env)

    say("")
    say("Starting the demo. It opens in your browser at "
        "http://127.0.0.1:7860")
    say("Keep this window open while presenting; press Ctrl+C to stop.")
    say("")
    try:
        return run([py, "app.py"] + app_args, env=env)
    except KeyboardInterrupt:
        say("")
        say("Demo stopped.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
