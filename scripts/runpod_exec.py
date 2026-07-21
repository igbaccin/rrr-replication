#!/usr/bin/env python3
"""
Run commands and download outputs from RunPod SSH endpoints that require a PTY.

Examples:
  python scripts/runpod_exec.py exec --target user@ssh.runpod.io --command "cd /root/RRR && tail -80 /tmp/rrr_t2.log"
  python scripts/runpod_exec.py download --target user@ssh.runpod.io --remote-dir /root/RRR/runs --local-root runs/downloaded
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import shlex
import socket
import sys
import tarfile
import time
import uuid
from pathlib import Path

try:
    import paramiko
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: paramiko. Install with `python -m pip install paramiko` "
        "or reinstall requirements."
    ) from exc


HOST_DEFAULT = "ssh.runpod.io"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\a]*(?:\a|\x1b\\)")


def _write(text: str) -> None:
    # Windows cmd / Git Bash default stdout to cp1252; the PTY stream from a
    # pod can contain Unicode (e.g. ollama pull progress block-chars). Write
    # via the buffered stdout in UTF-8 and replace any unencodable chars on the
    # terminal-side rather than crashing the whole exec call.
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.flush()
        except Exception:
            # Last resort: best-effort ASCII fallback, never crash the exec.
            sys.stdout.write(text.encode("ascii", errors="replace").decode("ascii"))
            sys.stdout.flush()


def _expand_path(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _default_key() -> str:
    env_key = _expand_path(os.environ.get("RUNPOD_SSH_KEY"))
    if env_key and os.path.exists(env_key):
        return env_key

    home_key = _expand_path("~/.ssh/id_ed25519")
    if home_key and os.path.exists(home_key):
        return home_key

    # v15.14: the silent %TEMP%\runpod_id_ed25519_codex fallback is gone — a
    # private key in a world-readable temp dir is bad practice, and using it
    # silently meant not knowing which identity authenticated.
    raise SystemExit("SSH key not found. Pass --key or set RUNPOD_SSH_KEY.")


def _parse_target(target: str | None) -> tuple[str, str]:
    target = target or os.environ.get("RUNPOD_SSH_TARGET")
    if not target:
        raise SystemExit("Missing target. Pass --target user@ssh.runpod.io or set RUNPOD_SSH_TARGET.")
    if "@" not in target:
        raise SystemExit("Target must look like user@ssh.runpod.io.")
    user, host = target.rsplit("@", 1)
    return user, host or HOST_DEFAULT


def _read_available(chan, seconds: float = 0.2) -> str:
    end = time.time() + seconds
    chunks: list[str] = []
    while time.time() < end:
        try:
            if chan.recv_ready():
                chunks.append(chan.recv(65535).decode("utf-8", errors="replace"))
            else:
                time.sleep(0.03)
        except socket.timeout:
            pass
    return "".join(chunks)


def _send_line(chan, line: str) -> None:
    chan.send(line + "\r")


_KNOWN_HOSTS_PATH = _expand_path("~/.ssh/known_hosts_runpod")


class _AcceptNewHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    """v15.14: accept-new with persistence (TOFU), replacing AutoAddPolicy.

    AutoAddPolicy accepted ANY key on EVERY connect with no persistence —
    full MITM exposure for each session. This policy stores the key on
    first contact in ~/.ssh/known_hosts_runpod; subsequent connections
    verify against it, so a changed key fails loudly (paramiko raises
    BadHostKeyException) instead of being silently accepted.
    """

    def missing_host_key(self, client, hostname, key):
        client.get_host_keys().add(hostname, key.get_name(), key)
        try:
            os.makedirs(os.path.dirname(_KNOWN_HOSTS_PATH), exist_ok=True)
            client.save_host_keys(_KNOWN_HOSTS_PATH)
        except OSError as e:
            _write(f"[runpod_exec] WARN: could not persist host key: {e}\n")
        _write(f"[runpod_exec] accepted new host key for {hostname} ({key.get_name()})\n")


class RunpodPty:
    def __init__(self, target: str, key_path: str, connect_timeout: int = 20):
        self.target = target
        self.key_path = key_path
        self.connect_timeout = connect_timeout

    def connect(self):
        user, host = _parse_target(self.target)
        client = paramiko.SSHClient()
        try:
            if _KNOWN_HOSTS_PATH and os.path.exists(_KNOWN_HOSTS_PATH):
                client.load_host_keys(_KNOWN_HOSTS_PATH)
        except Exception as e:
            _write(f"[runpod_exec] WARN: could not load {_KNOWN_HOSTS_PATH}: {e}\n")
        client.set_missing_host_key_policy(_AcceptNewHostKeyPolicy())
        try:
            client.connect(
                host,
                username=user,
                key_filename=self.key_path,
                look_for_keys=False,
                allow_agent=False,
                timeout=self.connect_timeout,
                banner_timeout=self.connect_timeout,
                auth_timeout=self.connect_timeout,
            )
        except paramiko.BadHostKeyException as exc:
            raise SystemExit(
                f"Host key for {host} CHANGED since it was first recorded in "
                f"{_KNOWN_HOSTS_PATH}. This is either a re-provisioned proxy "
                f"or a man-in-the-middle. If you expected the change, delete "
                f"that file and reconnect.\n{exc}"
            ) from exc
        return client

    def run_script(self, script: str, timeout: int, stream: bool = True) -> tuple[int, str]:
        marker = "__CODEX_DONE_%s__" % uuid.uuid4().hex
        remote_script = "/tmp/codex_run_%s.sh" % uuid.uuid4().hex
        client = self.connect()
        output = ""
        status: int | None = None
        try:
            chan = client.invoke_shell(term="xterm", width=180, height=60)
            chan.settimeout(1.0)
            output += _read_available(chan, 2)
            _send_line(chan, "stty -echo 2>/dev/null || true")
            output += _read_available(chan, 0.5)
            _send_line(chan, "cat > %s <<'CODEX_SH'" % shlex.quote(remote_script))
            for i in range(0, len(script), 4096):
                chan.send(script[i : i + 4096])
                time.sleep(0.002)
            if not script.endswith("\n"):
                chan.send("\n")
            chan.send("CODEX_SH\n")
            output += _read_available(chan, 0.5)
            _send_line(
                chan,
                "bash %s; status=$?; rm -f %s; echo %s:$status"
                % (shlex.quote(remote_script), shlex.quote(remote_script), marker),
            )

            deadline = time.time() + timeout
            seen = ""
            while time.time() < deadline:
                chunk = _read_available(chan, 0.5)
                if not chunk:
                    continue
                output += chunk
                clean = ANSI_RE.sub("", chunk)
                seen += clean
                if stream:
                    _write(clean)
                match = re.search(re.escape(marker) + r":(\d+)", seen)
                if match:
                    status = int(match.group(1))
                    break
            if status is None:
                output += "\n[TIMEOUT waiting for remote marker]\n"
                if stream:
                    _write("\n[TIMEOUT waiting for remote marker]\n")
                status = 124
            try:
                _send_line(chan, "exit")
            except Exception:
                pass
        finally:
            client.close()
        return status, ANSI_RE.sub("", output)


def _read_script(args) -> str:
    if args.command:
        return args.command
    if args.script_file:
        return Path(args.script_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Pass --command, --script-file, or pipe a shell script on stdin.")


def _extract_payload(output: str, begin: str, end: str) -> bytes:
    capture = False
    parts: list[str] = []
    for line in output.splitlines():
        if begin in line:
            capture = True
            parts.clear()
            continue
        if capture and end in line:
            payload = "".join(parts)
            if len(payload) < 32:
                raise SystemExit("Archive payload was empty or too small.")
            return base64.b64decode(payload)
        if capture:
            parts.append(re.sub(r"[^A-Za-z0-9+/=]", "", line))
    raise SystemExit("Archive markers were not found in a complete pair.")


def cmd_exec(args) -> int:
    key = _expand_path(args.key) or _default_key()
    runner = RunpodPty(args.target, key, connect_timeout=args.connect_timeout)
    status, output = runner.run_script(_read_script(args), timeout=args.timeout, stream=True)
    if args.log_file:
        Path(args.log_file).write_text(output, encoding="utf-8")
    return status


def cmd_download(args) -> int:
    key = _expand_path(args.key) or _default_key()
    target = args.target or os.environ.get("RUNPOD_SSH_TARGET")
    if not target:
        raise SystemExit("Missing target. Pass --target user@ssh.runpod.io or set RUNPOD_SSH_TARGET.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    local_root = Path(_expand_path(args.local_root) or "runs/downloaded")
    dest_dir = local_root / f"runpod_{stamp}"
    outputs_dir = dest_dir / "outputs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "runpod_outputs.tar.gz"

    begin = "__RRR_ARCHIVE_%s_BEGIN__" % uuid.uuid4().hex
    end = "__RRR_ARCHIVE_%s_END__" % uuid.uuid4().hex
    include_logs = args.include_log or []
    include_log_lines = "\n".join(
        "if [ -f {p} ]; then cp {p} \"$bundle\"/{name}; fi".format(
            p=shlex.quote(path),
            name=shlex.quote(Path(path).name),
        )
        for path in include_logs
    )

    script = f"""set -euo pipefail
remote_dir={shlex.quote(args.remote_dir)}
if [ ! -d "$remote_dir" ]; then
  echo "Remote output directory not found: $remote_dir" >&2
  exit 1
fi
bundle="$(mktemp -d)"
cleanup() {{ rm -rf "$bundle"; }}
trap cleanup EXIT
mkdir -p "$bundle/outputs"
cp -a "$remote_dir"/. "$bundle/outputs"/
{include_log_lines}
echo {shlex.quote(begin)}
tar -czf - -C "$bundle" . | base64
echo
echo {shlex.quote(end)}
"""

    runner = RunpodPty(target, key, connect_timeout=args.connect_timeout)
    status, output = runner.run_script(script, timeout=args.timeout, stream=False)
    if status != 0:
        _write(output[-4000:])
        return status

    archive_path.write_bytes(_extract_payload(output, begin, end))
    with tarfile.open(archive_path, "r:gz") as tf:
        try:
            tf.extractall(dest_dir, filter="data")
        except TypeError:
            tf.extractall(dest_dir)

    print("Downloaded RunPod outputs:")
    print(f"  Folder:  {outputs_dir}")
    print(f"  Archive: {archive_path}")
    for name in ("review_composed.md", "run_metrics.json", "citations.json", "review_ledger.json"):
        path = outputs_dir / name
        if path.exists():
            print(f"  {name}: {path}")

    if args.open and os.name == "nt":
        os.startfile(str(outputs_dir))  # type: ignore[attr-defined]
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod PTY executor and output downloader.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exec = sub.add_parser("exec", help="Run a shell script through an interactive PTY.")
    p_exec.add_argument("--target", default=os.environ.get("RUNPOD_SSH_TARGET"))
    p_exec.add_argument("--key", default=os.environ.get("RUNPOD_SSH_KEY"))
    p_exec.add_argument("--command", help="Shell command or script to run.")
    p_exec.add_argument("--script-file", help="Local shell script file to run remotely.")
    p_exec.add_argument("--timeout", type=int, default=300)
    p_exec.add_argument("--connect-timeout", type=int, default=20)
    p_exec.add_argument("--log-file", help="Write full PTY output to this file.")
    p_exec.set_defaults(func=cmd_exec)

    p_down = sub.add_parser("download", help="Download a remote directory into runs/downloaded.")
    p_down.add_argument("--target", default=os.environ.get("RUNPOD_SSH_TARGET"))
    p_down.add_argument("--key", default=os.environ.get("RUNPOD_SSH_KEY"))
    p_down.add_argument("--remote-dir", default="/root/RRR/runs")
    p_down.add_argument("--local-root", default="runs/downloaded")
    p_down.add_argument("--include-log", action="append", default=[])
    p_down.add_argument("--timeout", type=int, default=300)
    p_down.add_argument("--connect-timeout", type=int, default=20)
    p_down.add_argument("--open", action="store_true", help="Open the output folder on Windows.")
    p_down.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
