from __future__ import annotations

import argparse
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "market-master-component-classifier"


def promote(alias_from: str = "Staging", alias_to: str = "Production") -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    # Get the latest version with the source alias
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    src_version = None
    for v in versions:
        if hasattr(v, 'aliases') and alias_from in v.aliases:
            src_version = v.version
            break
    
    if not src_version:
        raise RuntimeError(f"No version tagged {alias_from}")
    
    # Set the target alias
    client.set_registered_model_alias(MODEL_NAME, alias_to, int(src_version))
    print(f"promoted: {MODEL_NAME} v{src_version} -> {alias_to}")


def rollback(target_version: Optional[int] = None) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    # Get current production version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    current_ver = None
    for v in versions:
        if hasattr(v, 'aliases') and 'Production' in v.aliases:
            current_ver = int(v.version)
            break
    versions = sorted(
        [int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")],
        reverse=True,
    )
    if target_version is None:
        # pick the latest version that is not the current Production
        for v in versions:
            if v != current_ver:
                target_version = v
                break
    if target_version is None:
        raise RuntimeError("No alternative version available to rollback")
    client.set_registered_model_alias(MODEL_NAME, "Production", int(target_version))
    print(f"rolled back: {MODEL_NAME} -> Production v{target_version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Registry utilities")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--promote", action="store_true", help="Promote Staging to Production"
    )
    g.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback Production to previous or specific version",
    )
    parser.add_argument(
        "--to-version",
        type=int,
        default=None,
        help="Explicit version to promote/rollback to",
    )
    args = parser.parse_args()

    if args.promote:
        if args.to_version is not None:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            client = MlflowClient()
            client.set_registered_model_alias(
                MODEL_NAME, "Production", int(args.to_version)
            )
            print(f"promoted explicit: {MODEL_NAME} -> Production v{args.to_version}")
        else:
            promote()
    elif args.rollback:
        rollback(args.to_version)


if __name__ == "__main__":
    main()
