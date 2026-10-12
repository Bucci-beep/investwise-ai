#!/usr/bin/env python3
"""Manual entry point. No default-app imports, automatic sign-in or inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
from session import (D5Session, DemoStopped, bootstrap, external_path,
                     private_directory, render)


def main():
    parser = argparse.ArgumentParser(description="Independent D5 fictional-text demo; the normal app stays unchanged.")
    parser.add_argument("action", choices=["login", "chat", "check-11"])
    parser.add_argument("--profile-dir", type=Path,
                        default=Path.home() / ".config/investwise/d5-chatgpt-plan")
    parser.add_argument("--output-dir", type=Path,
                        help="A new directory outside the repository for private local session records")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or os.name != "posix":
        parser.error("Use Python 3.12 on macOS or Linux (the existing OAuth lock uses POSIX).")
    os.umask(0o077)
    bootstrap()
    from ai.d5_chatgpt_auth import AuthBlocked, ChatGPTCredentials
    try:
        credentials = ChatGPTCredentials(external_path(args.profile_dir))
        if args.action == "login":
            credentials.authorise()
            credentials.list_models()
            print("Sign-in completed and gpt-6.1-sol is listed. Credentials remain outside the repository.")
            return 0
        output = args.output_dir or (Path.home() / ".local/state/investwise/d5-demo" /
            (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]))
        output = private_directory(output)
        if args.action == "check-11":
            from check_eleven import run
            return run(credentials, output)
        session = D5Session(credentials, output)
        print("InvestWise — independent D5 example | GPT-6.1 Sol, low reasoning")
        print("Use fictional text only. Replies and safe backend traces are recorded locally at:", output)
        print("The normal app is unchanged. This example cannot save a financial profile.")
        print("Commands: /help /confirm /change /pause /resume /stop /new /exit")
        print("\n" + session.question)
        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nD5 example closed.")
                return 0
            if not text:
                continue
            if text == "/exit":
                return 0
            before = session.backend.calls
            result = session.command(text) if text.startswith("/") else session.submit(text)
            render(result)
            print(f"[Interpretation calls for this action: {session.backend.calls - before}]")
            if text == "/new":
                print(session.question)
    except AuthBlocked as error:
        print("Sign-in/access stopped:", error.code, file=sys.stderr)
        print("No automatic retry or model substitution. Use the documented login command if needed.", file=sys.stderr)
        return 2
    except DemoStopped as error:
        print(str(error), file=sys.stderr)
        print("No retry or fallback. Inspect the private safe receipt for the recorded stage/code.", file=sys.stderr)
        return 2
    except Exception:
        # Provider exception messages and stack traces must never expose tokens.
        print("D5 could not start or continue. Check Python, installed dependencies and external directory settings.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
