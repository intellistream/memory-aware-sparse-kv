"""Download and seal the pinned GLM-5.3 tiny engineering model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID = "inference-optimization/GLM-5.3-0.6B-A0.4B"
REVISION = "20aae340d157bd2215b9d81165be87e94686dfdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_and_seal(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise RuntimeError("Downloaded snapshot lacks config.json or tokenizer.json")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "glm_moe_dsa":
        raise RuntimeError(f"Unexpected model_type: {config.get('model_type')!r}")
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise RuntimeError(f"Unexpected architecture: {config.get('architectures')!r}")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("Downloaded snapshot has no safetensors weights")
    artifacts = [config_path, tokenizer_path, *weights]
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "files": {
            path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        },
    }
    temporary = model_dir / ".m0a_model_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(model_dir / ".m0a_model_manifest.json")
    revision_tmp = model_dir / ".m0a_revision.tmp"
    revision_tmp.write_text(REVISION + "\n")
    revision_tmp.replace(model_dir / ".m0a_revision")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("/workspace/memecho/models/GLM-5.3-0.6B-A0.4B"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    marker = args.output / ".m0a_revision"
    if marker.exists() and marker.read_text().strip() not in {"", REVISION}:
        raise RuntimeError("Refusing a model directory sealed at another revision")
    sealed = (
        marker.is_file()
        and marker.read_text().strip() == REVISION
        and (args.output / "config.json").is_file()
        and (args.output / "tokenizer.json").is_file()
        and any(args.output.glob("*.safetensors"))
    )
    if not args.verify_only and not sealed:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to download the model") from error
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=str(args.output),
        )
    result = verify_and_seal(args.output)
    print(json.dumps({
        "model_id": result["model_id"],
        "revision": result["revision"],
        "files": len(result["files"]),
        "bytes": sum(item["size"] for item in result["files"].values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
