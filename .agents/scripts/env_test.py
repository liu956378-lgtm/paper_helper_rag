import os
import sys
import shutil
import json
import subprocess


def get_command_output(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


def main():

    result = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },

        "environment": {
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },

        "commands": {
            "python": shutil.which("python"),
            "pip": shutil.which("pip"),
            "conda": shutil.which("conda"),
        },
    }

    result["python"]["is_virtual_environment"] = (
        sys.prefix != sys.base_prefix
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()