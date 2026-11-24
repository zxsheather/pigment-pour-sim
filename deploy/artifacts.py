"""Pinned source checkout and verified release/base checkpoint installation.

All paths are relative to this checkout by default; no original machine paths.
This module uses only the standard library until downloading the public base.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'deploy/model_manifest.json'
WALLX_REVISION = 'ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_file(path, spec, full=True):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        raise ValueError(f'Missing file or wrong size: {path.name}')
    if full and digest(path) != spec['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')


def source_checkout(destination):
    destination = Path(destination).resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        bundled = ROOT / 'deploy/upstream/wall-x-ce2a1ae.bundle'
        source = str(bundled) if bundled.is_file() else 'https://github.com/X-Square-Robot/wall-x.git'
        subprocess.run(['git', '-c', 'http.version=HTTP/1.1', 'clone', '--no-checkout',
                        source, str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', WALLX_REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != WALLX_REVISION:
        raise ValueError('Wall-X revision differs from tested pin; use another empty destination.')
    subprocess.run([os.sys.executable, str(ROOT / 'experiments/grasp/patch_wallx_runtime.py'),
                    str(destination)], check=True)
    return destination


def verify_checkpoint(destination, full=True):
    manifest = json.loads(MANIFEST.read_text())
    for name, spec in manifest['files'].items():
        # --quick skips only the 8.33 GB base hash; overlay and all metadata still hash.
        check_file(Path(destination) / name, spec, full=full or name != 'model.safetensors')
    print(json.dumps({'verified': True, 'base_sha256_checked': full,
                      'overlay_sha256': manifest['files']['action_expert.safetensors']['sha256']}))


def extract_overlay(archive, destination, manifest):
    """Never extract archive links, nested paths, devices or unlisted files."""
    expected = set(manifest['files']) - {'model.safetensors'}
    seen = set()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
            spec = manifest['files'][member.name]
            if member.size != spec['bytes']:
                raise ValueError('Archive member size mismatch')
            target = destination / member.name
            if target.is_symlink():
                raise ValueError('Refusing to overwrite a symlink')
            temp = target.with_name(target.name + '.partial')
            if temp.is_symlink():
                raise ValueError('Refusing a symlink temporary file')
            with tar.extractfile(member) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            check_file(temp, spec)
            temp.replace(target)
            seen.add(member.name)
    if seen != expected:
        raise ValueError(f'Incomplete archive: {sorted(expected - seen)}')


def download(destination, archive=None):
    manifest = json.loads(MANIFEST.read_text())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    spec = manifest['release_asset']
    if archive is None:
        archive = ROOT / 'deploy/artifacts' / spec['name']
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release_tag']}/{spec['name']}"
            partial = archive.with_name(archive.name + '.partial')
            print(f'Downloading {spec["name"]} ...', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'pigment-pour-sim-deploy'})
            with urllib.request.urlopen(request, timeout=120) as src, partial.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            check_file(partial, spec)
            partial.replace(archive)
    check_file(archive, spec)
    extract_overlay(archive, destination, manifest)
    # Only the frozen base tensor file comes from HF. Overlay tokenizer/configs
    # and normalizers must NOT be overwritten by the upstream base snapshot.
    target = destination / 'model.safetensors'
    if not target.exists():
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=manifest['base']['repository'],
                                     revision=manifest['base']['revision'], filename=target.name)
        check_file(downloaded, manifest['files'][target.name])
        # A symlink avoids another 8.33 GB copy of the HF cache on the target PC.
        target.symlink_to(Path(downloaded).resolve())
    verify_checkpoint(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['source', 'download', 'verify'])
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'deploy/checkpoint')
    parser.add_argument('--wallx', type=Path, default=ROOT / 'deploy/vendor/wall-x')
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded release asset')
    parser.add_argument('--quick', action='store_true', help='Verify all except full base SHA256')
    args = parser.parse_args()
    if args.command == 'source':
        source_checkout(args.wallx)
    elif args.command == 'download':
        download(args.checkpoint, args.archive)
    else:
        verify_checkpoint(args.checkpoint, not args.quick)


if __name__ == '__main__':
    main()
