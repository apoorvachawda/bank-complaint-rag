"""Create (if needed) and push this app to a Hugging Face Space.

Requires HF_TOKEN in the environment (a write token). Uploads the code, the
Space card, requirements.txt, and sets the runtime variables. Secrets
(API keys) must be set once in the Space UI — this script does NOT upload
.env, so no credentials ever leave the machine through here.

Usage:
    HF_TOKEN=hf_xxx python -m deploy.push_to_space --space <user>/<name>
"""

import argparse
import os

from huggingface_hub import HfApi

# Files the running Space needs. Deliberately excludes data/, .env, chroma_db/,
# the BM25 pickle (dense-only deployment reads Qdrant Cloud).
INCLUDE = [
    "app.py",
    "requirements.txt",
    "pyproject.toml",
    "src/",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="<user>/<space-name>")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    api.create_repo(
        repo_id=args.space,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )

    # Space card (README.md with HF frontmatter) drives Space config.
    api.upload_file(
        path_or_fileobj="deploy/README_SPACE.md",
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
    )

    for path in INCLUDE:
        if path.endswith("/"):
            api.upload_folder(
                folder_path=path.rstrip("/"),
                path_in_repo=path.rstrip("/"),
                repo_id=args.space,
                repo_type="space",
                ignore_patterns=["__pycache__/*", "*.pyc"],
            )
        else:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path,
                repo_id=args.space,
                repo_type="space",
            )

    # Non-secret runtime config.
    for key, value in {"VECTOR_BACKEND": "qdrant", "RETRIEVER_NAME": "dense"}.items():
        api.add_space_variable(repo_id=args.space, key=key, value=value)

    print(f"Pushed to https://huggingface.co/spaces/{args.space}")
    print(
        "Now set secrets (GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, LANGFUSE_*) "
        "in the Space Settings UI, then it will build."
    )


if __name__ == "__main__":
    main()
