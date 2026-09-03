#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

MODELS = (
    "Eureka-Audio",
    "HY-MT",
    "Qwen3-ASR",
    "Qwen3-TTS",
    "QwenLLM",
    "chatTTS",
    "moonshine",
    "sensevoice",
    "vits-melo-tts-zh_en",
    "whisper",
    "zipformer",
)
SKIP_PARTS = {
    ".git",
    "models",
    "build",
    "build-aarch64",
    "build-aarch64-v2",
    "build-host",
    "outputs",
    "test_outputs",
    "test_data",
    "calib_data",
    "LLM-TPU",
    "tmp",
    "__pycache__",
}
FORBIDDEN = re.compile(
    r"/home/|/mnt/|172\.16\."
    r"|sshpass\s+-p\s+(?![\"']?(?:\$\{|\$[A-Za-z_]|<\w+>)[\"']?)\S+"
    r"|(?:BOARD_)?PASS\s*=\s*(?![\"']?(?:\$\{|\$[A-Za-z_]|<\w+>)[\"']?)[^\s#]+"
)
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def tracked_files(root):
    """返回 git 已跟踪的 repo-相对 POSIX 路径集合（用于避免扫描被忽略的生成产物）。"""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return {path for path in result.stdout.split("\0") if path}


def rel(path, root):
    return path.relative_to(root).as_posix()


def report(kind, path, message):
    print(f"{kind}: {path}: {message}")


def report_limited(kind, paths, root, message, limit=8):
    for path in paths[:limit]:
        report(kind, rel(path, root), message)
    if len(paths) > limit:
        report(kind, "...", f"同类问题还有 {len(paths) - limit} 个")
    return len(paths)


def check_layout(root, tracked):
    errors = 0
    for name in MODELS:
        model = root / name
        required = [
            "README.md",
            "requirements.txt",
            ".gitignore",
            "cpp/CMakeLists.txt",
            "cpp/build.sh",
            "models",
        ]
        required.append("test_audios" if name == "Eureka-Audio" else "test_data")
        required.append("scripts" if name == "QwenLLM" else "python")
        for item in required:
            path = model / item
            if not path.exists():
                report("ERROR", rel(path, root), "缺少统一目录或入口")
                errors += 1
        if not (model / "deploy_to_board.sh").exists() and name != "QwenLLM":
            report("ERROR", name, "缺少 deploy_to_board.sh")
            errors += 1

        onnx_bad = []
        onnx = model / "models" / "onnx"
        if onnx.exists():
            for path in onnx.rglob("*"):
                rel_path = rel(path, root)
                if (
                    path.is_file()
                    and rel_path in tracked
                    and path.name != ".gitkeep"
                    and path.suffix not in {".onnx", ".data"}
                ):
                    onnx_bad.append(path)
        errors += report_limited("ERROR", onnx_bad, root, "models/onnx 只能包含 ONNX 及其 external data")

        bm_bad = []
        bm = model / "models" / "BM1684X"
        if bm.exists():
            for path in bm.rglob("*"):
                rel_path = rel(path, root)
                if (
                    path.is_file()
                    and rel_path in tracked
                    and path.name != ".gitkeep"
                    and path.suffix not in {".bmodel", ".txt"}
                ):
                    bm_bad.append(path)
        errors += report_limited("ERROR", bm_bad, root, "models/BM1684X 只能包含最终 bmodel")
    return errors


def check_links(root, tracked):
    errors = 0
    for rel_path in tracked:
        if not rel_path.endswith(".md"):
            continue
        if any(part in SKIP_PARTS for part in Path(rel_path).parts):
            continue
        doc = root / rel_path
        for target in LINK.findall(doc.read_text(errors="ignore")):
            target = target.split("#", 1)[0].strip()
            if (
                not target
                or target.startswith(("http://", "https://", "mailto:"))
                or any(char.isspace() for char in target)
                or any(symbol in target for symbol in ("→", "←"))
            ):
                continue
            if not (doc.parent / target).exists():
                report("ERROR", rel(doc, root), f"内部链接不存在: {target}")
                errors += 1
    return errors


def check_sources(root, tracked):
    errors = 0
    for rel_path in tracked:
        if rel_path == rel(Path(__file__).resolve(), root):
            continue
        if any(part in SKIP_PARTS for part in Path(rel_path).parts):
            continue
        path = root / rel_path
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".sh", ".md", ".cmake"} and path.name != "CMakeLists.txt":
            continue
        for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if FORBIDDEN.search(line):
                report("ERROR", f"{rel(path, root)}:{line_no}", "包含本机路径、固定板卡地址或硬编码认证信息")
                errors += 1
    return errors


def check_shell(root, tracked):
    errors = 0
    for rel_path in tracked:
        if not rel_path.endswith(".sh"):
            continue
        if any(part in SKIP_PARTS for part in Path(rel_path).parts):
            continue
        path = root / rel_path
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            report("ERROR", rel(path, root), result.stderr.strip() or "Shell 语法错误")
            errors += 1
    return errors


# 根目录散落编译产物（无模型归属的输出，须全部收进各模型 compile/tmp 或 models/）
_ROOT_STRAY_DIRS = {"models", "compile", "tmp", "build", "calib_data", "outputs", "results"}
_ROOT_STRAY_FILE_RE = re.compile(
    r"\.(npz|mlir|prototxt|modify|bmodel\.net_0\.profile|bmodel\.json)$"
    r"|\.layer_group_config\.json$|\.layer_group_cache\.json$|\.ref_files\.json$"
    r"|(^|_)(encoder|decoder|joiner)_[a-z0-9_]+weight\.npz$"
)
_COMPILE_OUTPUT_RE = re.compile(
    r"\.(npz|mlir|prototxt|modify|profile)$"
    r"|\.layer_group_config\.json$|\.layer_group_cache\.json$|\.ref_files\.json$"
    r"|^compiler_profile_.*\.txt$"
)


def check_root_outputs(root):
    """仓库根目录不得出现模型编译散落产物。"""
    errors = 0
    legal_dirs = {"0_Toolkits", "1_third_party", "2_utils", "3_docker", "4_tools", ".claude", ".git"}
    for entry in sorted(root.iterdir()):
        if entry.name in legal_dirs:
            continue
        if entry.is_dir():
            if entry.name in _ROOT_STRAY_DIRS or entry.name.lower().startswith("models"):
                report("ERROR", entry.name, "根目录散落输出目录，应并入对应模型目录内")
                errors += 1
        elif _ROOT_STRAY_FILE_RE.search(entry.name):
            report("ERROR", entry.name, "根目录散落编译产物，应清理或收进 compile/tmp")
            errors += 1
    return errors


def check_disk_outputs(root):
    """检查被 gitignore 的模型输出目录，发现编译副产物和错误层级。"""
    errors = 0
    for name in MODELS:
        model = root / name
        onnx = model / "models" / "onnx"
        if onnx.exists():
            bad = []
            for path in onnx.rglob("*"):
                if not path.is_file() or path.name == ".gitkeep":
                    continue
                if path.suffix in {".onnx", ".data"}:
                    continue
                if name == "vits-melo-tts-zh_en" and (
                    path.name == "LICENSE"
                    or path.suffix in {".txt", ".fst", ".utf8", ".md"}
                ):
                    continue
                bad.append(path)
            errors += report_limited("ERROR", bad, root, "models/onnx 含非模型输入文件")

        bm = model / "models" / "BM1684X"
        if bm.exists():
            bad = []
            for path in bm.rglob("*"):
                relative = path.relative_to(bm).parts
                if name == "HY-MT" and len(relative) >= 2 and relative[1] == "config":
                    continue
                if path.is_dir():
                    if name == "HY-MT" and len(relative) == 1:
                        continue
                    bad.append(path)
                elif (
                    path.name != ".gitkeep"
                    and path.suffix not in {".bmodel", ".txt"}
                    and not (name == "whisper" and path.suffix == ".npy")
                ):
                    bad.append(path)
            errors += report_limited("ERROR", bad, root, "models/BM1684X 含编译副产物或错误层级")

        compile_dir = model / "compile"
        if compile_dir.exists():
            bad = [
                path for path in compile_dir.iterdir()
                if path.is_file() and _COMPILE_OUTPUT_RE.search(path.name)
            ]
            errors += report_limited("ERROR", bad, root, "compile 根目录含中间产物，应移入临时目录或清理")
    return errors


def main():
    parser = argparse.ArgumentParser(description="检查 Sophon Model Zoo 的目录和交付约定")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--disk",
        action="store_true",
        help="额外检查被 gitignore 的模型输出和编译副产物",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    tracked = tracked_files(root)
    if tracked is None:
        print("错误: 无法获取 git 跟踪文件列表，请确认在 git 仓库内运行", file=sys.stderr)
        return 2
    errors = check_layout(root, tracked)
    errors += check_links(root, tracked)
    errors += check_sources(root, tracked)
    errors += check_shell(root, tracked)
    errors += check_root_outputs(root)
    if args.disk:
        errors += check_disk_outputs(root)
    print(f"检查完成: {errors} 个问题")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
