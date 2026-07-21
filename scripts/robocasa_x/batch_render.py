"""Render all raw demo.hdf5 files below a RoboCasa-X dataset root in parallel."""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

CAMERAS = {
    "panda": "panda_agentview_left",
    "panda-og": "panda_agentview_left",
    "jaco": "jaco_agentview_left",
    "iiwa": "iiwa_agentview_left",
    "ur5e": "ur5e_agentview_left",
    "kinova3": "kinova_agentview_left",
}


def render_one(source: Path, args: argparse.Namespace) -> Tuple[Path, int, str]:
    script_path = Path(__file__).with_name("render_dataset.py")
    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--dataset",
        str(source),
        "--camera",
        args.camera,
        "--wrist-camera",
        args.wrist_camera,
        "--output-name",
        args.output_name,
    ]
    if args.max_demos is not None:
        command.extend(["--max-demos", str(args.max_demos)])
    if args.no_compress:
        command.append("--no-compress")
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return source, process.returncode, process.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing raw demo.hdf5 files.")
    parser.add_argument("--robot", choices=sorted(CAMERAS), required=True)
    parser.add_argument("--camera", help="Override the robot-relative third-person camera.")
    parser.add_argument("--wrist-camera", default="robot0_eye_in_hand")
    parser.add_argument("--output-name", default="demo_gentex_im320.hdf5")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.camera = args.camera or CAMERAS[args.robot]
    sources = []
    for source in sorted(args.root.rglob("demo.hdf5")):
        if args.overwrite or not source.with_name(args.output_name).exists():
            sources.append(source)
    if not sources:
        print("No unrendered demo.hdf5 files found.")
        return
    print(f"Rendering {len(sources)} files with camera {args.camera} on {args.workers} workers")
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_one, source, args) for source in sources]
        for future in as_completed(futures):
            source, returncode, output = future.result()
            prefix = f"[{source.parent.name}]"
            print("".join(f"{prefix} {line}\n" for line in output.splitlines()), end="")
            if returncode:
                failures.append(str(source))
    if failures:
        raise RuntimeError(f"Rendering failed for {len(failures)} files: {failures}")


if __name__ == "__main__":
    main()
