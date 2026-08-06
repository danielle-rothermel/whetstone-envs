import shutil
import subprocess
import tomllib
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from zipfile import ZipFile

REPOSITORY_ROOT = Path(__file__).parents[1]
LICENSE_FILES = [
    "LICENSE",
    "src/whetstone_envs/c23/attribution/LICENSE",
    "src/whetstone_envs/c23/attribution/PROVENANCE.md",
]


def test_distribution_declares_every_included_license(tmp_path: Path) -> None:
    metadata = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(),
    )["project"]

    assert metadata["license"] == "MIT AND Apache-2.0"
    assert metadata["license-files"] == LICENSE_FILES
    assert not any(
        classifier.startswith("License ::")
        for classifier in metadata["classifiers"]
    )
    assert all((REPOSITORY_ROOT / path).is_file() for path in LICENSE_FILES)

    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(  # noqa: S603 - fixed package build command
        [
            uv,
            "build",
            "--wheel",
            "--no-sources",
            "--out-dir",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel_path,) = tmp_path.glob("*.whl")
    with ZipFile(wheel_path) as wheel:
        metadata_path = next(
            name
            for name in wheel.namelist()
            if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = BytesParser(policy=default).parsebytes(
            wheel.read(metadata_path),
        )

    assert wheel_metadata["License-Expression"] == "MIT AND Apache-2.0"
    assert wheel_metadata.get_all("License-File") == LICENSE_FILES
