"""Lossless VCDIFF archives for immutable, unselected checkpoint weight files.

The shared base must remain intact. Decode to a streaming SHA256 before an
explicit --remove-original can unlink anything. Sidecar configs stay in place.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_dir(path.parent)


def tool_env():
    env = os.environ.copy()
    env.pop("XDELTA", None)
    return env


def decode_hash(tool, base, patch, output=None):
    digest = hashlib.sha256()
    size = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(tool), "-d", "-c", "-D", "-R", "-s", str(base), str(patch)],
                                   stdout=subprocess.PIPE, stderr=errors, env=tool_env())
        try:
            for chunk in iter(lambda: process.stdout.read(4 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
                if output is not None:
                    output.write(chunk)
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError(f"xdelta decode failed ({code}): {errors.read().decode(errors='replace')}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    return digest.hexdigest(), size


def archive(args):
    checkpoint = args.checkpoint.resolve()
    original = checkpoint / "model.safetensors"
    base = args.base.resolve()
    require(original.is_file() and not original.is_symlink(), "Expected a regular model weight file")
    require(base.is_file() and not original.samefile(base), "A base cannot archive itself")
    destination = checkpoint / "model.safetensors.xdelta-archive"
    partial = destination.with_name(destination.name + ".partial")
    require(not destination.exists() and not partial.exists(), "Archive path already exists")
    original_id, base_id = identity(original), identity(base)
    require(shutil.disk_usage(checkpoint).free > original_id[1] * 1.2, "Insufficient temporary space")
    original_hash, base_hash = sha256(original), sha256(base)
    partial.mkdir()
    patch = partial / "weights.vcdiff"
    subprocess.run([str(args.xdelta), "-e", "-1", "-B", "67108864", "-W", "8388608",
                    "-S", "none", "-D", "-s", str(base), str(original), str(patch)],
                   check=True, env=tool_env())
    require(patch.stat().st_size < original_id[1], "Archive did not save space; original retained")
    reconstructed_hash, reconstructed_size = decode_hash(args.xdelta, base, patch)
    require((reconstructed_hash, reconstructed_size) == (original_hash, original_id[1]), "Decoded original hash/size mismatch")
    require(identity(original) == original_id and identity(base) == base_id, "Input changed during verification")
    version = subprocess.run([str(args.xdelta), "-V"], capture_output=True, text=True, env=tool_env())
    manifest = dict(format="xdelta3-vcdiff-v1", verified=True,
                    original_file=str(original), original_sha256=original_hash, original_size=original_id[1],
                    base_file=str(base), base_sha256=base_hash, base_size=base_id[1],
                    patch_sha256=sha256(patch), patch_size=patch.stat().st_size,
                    streamed_reconstruction_sha256=reconstructed_hash,
                    original_removed=False, tool=str(args.xdelta), tool_sha256=sha256(args.xdelta),
                    tool_version=(version.stdout+version.stderr).splitlines()[0])
    with patch.open("rb") as stream:
        os.fsync(stream.fileno())
    restore_command = [".venv-sim/bin/python", str(Path(__file__).resolve()), "restore",
                       "--archive", str(destination), "--xdelta", str(args.xdelta), "--output", str(original)]
    (partial / "RESTORE.txt").write_text(
        "Lossless old checkpoint archive. Keep the shared base listed in manifest.json.\n"
        + " ".join(shlex.quote(value) for value in restore_command) + "\n"
        + "Restoration checks the base, patch and complete original SHA256; configs remain beside this archive.\n")
    write_json(partial / "manifest.json", manifest)
    partial.replace(destination)
    sync_dir(checkpoint)
    if args.remove_original:
        require(identity(original) == original_id and identity(base) == base_id, "Input changed before removal")
        original.unlink()
        sync_dir(checkpoint)
        manifest["original_removed"] = True
        write_json(destination / "manifest.json", manifest)
    print(json.dumps(dict(archive=str(destination), **manifest)), flush=True)


def restore(args):
    manifest = json.loads((args.archive / "manifest.json").read_text())
    require(manifest["format"] == "xdelta3-vcdiff-v1" and manifest["verified"], "Unverified archive")
    base = args.base.resolve() if args.base else Path(manifest["base_file"])
    patch = args.archive / "weights.vcdiff"
    require(sha256(base) == manifest["base_sha256"], "Shared base hash mismatch")
    require(sha256(patch) == manifest["patch_sha256"], "Patch hash mismatch")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".restoring")
    require(not output.exists() and not partial.exists(), "Refusing to overwrite an existing file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as stream:
        restored_hash, restored_size = decode_hash(args.xdelta, base, patch, stream)
        require((restored_hash, restored_size) == (manifest["original_sha256"], manifest["original_size"]), "Restored original hash/size mismatch")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output)
    sync_dir(output.parent)
    print(json.dumps(dict(restored=str(output), sha256=restored_hash, bytes=restored_size)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("archive")
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--base", type=Path, required=True)
    encode.add_argument("--remove-original", action="store_true")
    decode = commands.add_parser("restore")
    decode.add_argument("--archive", type=Path, required=True)
    decode.add_argument("--base", type=Path)
    decode.add_argument("--output", type=Path, required=True)
    for command in (encode, decode):
        command.add_argument("--xdelta", type=Path, required=True)
    args = parser.parse_args()
    args.xdelta = args.xdelta.resolve()
    require(args.xdelta.is_file(), "xdelta3 binary not found")
    archive(args) if args.command == "archive" else restore(args)


if __name__ == "__main__":
    main()
