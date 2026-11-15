"""Fetch pinned public dependencies; keep them outside this repository's Git index."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd):
    subprocess.run(list(args), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Dependency names; empty means all")
    args = parser.parse_args()
    sources = json.loads((ROOT/"sources.json").read_text())
    for name in args.names or sources:
        if name not in sources:
            raise ValueError(f"Unknown source {name!r}; choose from {list(sources)}")
        spec = sources[name]
        dest = ROOT/spec["destination"]
        marker = dest/".source-revision"
        if marker.exists() and marker.read_text().strip() == spec["commit"]:
            if all((dest/p).exists() for p in spec["required"]):
                print(f"Ready: {name} @ {spec['commit']}")
                continue
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError(f"Existing unverified directory: {dest}. Inspect it before replacing.")
        dest.mkdir(parents=True, exist_ok=True)
        run("git", "init", "-q", cwd=dest)
        run("git", "remote", "add", "origin", spec["url"], cwd=dest)
        if spec.get("sparse_paths"):
            run("git", "sparse-checkout", "init", "--cone", cwd=dest)
            run("git", "sparse-checkout", "set", *spec["sparse_paths"], cwd=dest)
        run("git", "fetch", "--depth", "1", "origin", spec["commit"], cwd=dest)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=dest)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        if actual != spec["commit"]:
            raise RuntimeError(f"Revision mismatch for {name}: {actual}")
        marker.write_text(actual+"\n")
        print(f"Fetched: {name} @ {actual}")


if __name__ == "__main__":
    main()
