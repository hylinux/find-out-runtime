from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from azure.core import AzureClouds
from azure.core.credentials import TokenCredential
from azure.identity import AzureAuthorityHosts, InteractiveBrowserCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
from azure.mgmt.databricks.models import Workspace
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.credentials_provider import CredentialsProvider, CredentialsStrategy
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="find-out-runtime",
    help="Find Azure Databricks clusters and jobs using a specified DBR runtime.",
    no_args_is_help=True,
)
console = Console()

DATABRICKS_SCOPE = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"


class AzureCloud(str, Enum):
    GLOBAL = "global"
    CHINA = "china"


@dataclass(frozen=True)
class CloudConfig:
    name: AzureCloud
    authority: str
    arm_cloud: AzureClouds
    arm_scope: str


CLOUD_CONFIGS: dict[AzureCloud, CloudConfig] = {
    AzureCloud.GLOBAL: CloudConfig(
        name=AzureCloud.GLOBAL,
        authority=AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
        arm_cloud=AzureClouds.AZURE_PUBLIC_CLOUD,
        arm_scope="https://management.azure.com/.default",
    ),
    AzureCloud.CHINA: CloudConfig(
        name=AzureCloud.CHINA,
        authority=AzureAuthorityHosts.AZURE_CHINA,
        arm_cloud=AzureClouds.AZURE_CHINA_CLOUD,
        arm_scope="https://management.chinacloudapi.cn/.default",
    ),
}


@dataclass(frozen=True)
class RuntimeUsage:
    subscription_id: str
    cloud: str
    resource_group: str
    workspace_name: str
    workspace_url: str
    resource_type: str
    resource_name: str
    resource_id: str
    runtime: str
    detail: str | None = None


def create_credential(cloud_config: CloudConfig) -> InteractiveBrowserCredential:
    """Create a browser credential using the default organizations tenant."""
    return InteractiveBrowserCredential(
        authority=cloud_config.authority,
        timeout=600,
    )


def preauthenticate(
    credential: InteractiveBrowserCredential,
    cloud_config: CloudConfig,
) -> None:
    """Request ARM and Databricks tokens before scanning."""
    console.print()
    console.print("[bold]Opening Microsoft Entra sign-in in your browser...[/]")
    credential.get_token(cloud_config.arm_scope)
    credential.get_token(DATABRICKS_SCOPE)
    console.print("[green]Authentication succeeded.[/]")
    console.print()


def runtime_matches(spark_version: str | None, target_runtime: str) -> bool:
    if not spark_version:
        return False

    target = target_runtime.strip().rstrip(".")
    if not target:
        return False

    return (
        spark_version == target  # noqa: PIE810
        or spark_version.startswith(f"{target}.")
        or spark_version.startswith(f"{target}-")
    )


def get_resource_group(resource_id: str | None) -> str:
    if not resource_id:
        return "unknown"

    parts = resource_id.strip("/").split("/")
    for index, part in enumerate(parts):
        if part.lower() == "resourcegroups" and index + 1 < len(parts):
            return parts[index + 1]

    return "unknown"


def normalize_workspace_host(workspace_url: str) -> str:
    host = workspace_url.rstrip("/")
    return host if host.startswith("https://") else f"https://{host}"


def get_workspace_url(workspace: Workspace) -> str | None:
    properties = workspace.properties
    return properties.workspace_url if properties is not None else None


def list_workspaces(
    subscription_id: str,
    credential: TokenCredential,
    cloud_config: CloudConfig,
) -> list[Workspace]:
    client = AzureDatabricksManagementClient(
        credential=credential,
        subscription_id=subscription_id,
        cloud_setting=cloud_config.arm_cloud,
    )
    return list(client.workspaces.list_by_subscription())


class InteractiveBrowserCredentialsStrategy(CredentialsStrategy):
    """Adapt Azure Identity interactive authentication to Databricks SDK."""

    def __init__(self, credential: InteractiveBrowserCredential) -> None:
        self._credential = credential

    def auth_type(self) -> str:
        return "interactive-browser-entra"

    def __call__(self, cfg: Config) -> CredentialsProvider:
        del cfg

        def credentials_provider() -> dict[str, str]:
            token = self._credential.get_token(DATABRICKS_SCOPE)
            return {"Authorization": f"Bearer {token.token}"}

        return credentials_provider


def create_workspace_client(
    workspace: Workspace,
    credential: InteractiveBrowserCredential,
) -> WorkspaceClient:
    workspace_url = get_workspace_url(
        workspace
    )

    if not workspace_url:
        raise RuntimeError(
            "Workspace does not contain workspace_url."
        )

    strategy = InteractiveBrowserCredentialsStrategy(
        credential
    )

    return WorkspaceClient(
        host=normalize_workspace_host(
            workspace_url
        ),
        credentials_strategy=strategy,
    )


def scan_clusters(
    client: WorkspaceClient,
    subscription_id: str,
    cloud_config: CloudConfig,
    workspace_name: str,
    workspace_url: str,
    resource_group: str,
    target_runtime: str,
) -> Iterator[RuntimeUsage]:
    for cluster in client.clusters.list():
        spark_version = cluster.spark_version
        if not runtime_matches(spark_version, target_runtime):
            continue

        state = cluster.state
        detail = getattr(state, "value", str(state)) if state is not None else None

        yield RuntimeUsage(
            subscription_id=subscription_id,
            cloud=cloud_config.name.value,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_url=workspace_url,
            resource_type="CLUSTER",
            resource_name=cluster.cluster_name or cluster.cluster_id or "unknown",
            resource_id=cluster.cluster_id or "",
            runtime=spark_version or "",
            detail=detail,
        )


def scan_jobs(
    client: WorkspaceClient,
    subscription_id: str,
    cloud_config: CloudConfig,
    workspace_name: str,
    workspace_url: str,
    resource_group: str,
    target_runtime: str,
) -> Iterator[RuntimeUsage]:
    for job_summary in client.jobs.list():
        job_id = job_summary.job_id
        if job_id is None:
            continue

        try:
            job = client.jobs.get(job_id=job_id)
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]Warning:[/] Unable to read job {job_id}: {exc}"
            )
            continue

        settings = job.settings
        if settings is None:
            continue

        job_name = settings.name or f"job-{job_id}"

        for task in settings.tasks or []:
            new_cluster = task.new_cluster
            if new_cluster is None:
                continue

            spark_version = new_cluster.spark_version
            if not runtime_matches(spark_version, target_runtime):
                continue

            yield RuntimeUsage(
                subscription_id=subscription_id,
                cloud=cloud_config.name.value,
                resource_group=resource_group,
                workspace_name=workspace_name,
                workspace_url=workspace_url,
                resource_type="JOB_TASK_CLUSTER",
                resource_name=job_name,
                resource_id=str(job_id),
                runtime=spark_version or "",
                detail=f"task={task.task_key}",
            )

        for job_cluster in settings.job_clusters or []:
            new_cluster = job_cluster.new_cluster
            if new_cluster is None:
                continue

            spark_version = new_cluster.spark_version
            if not runtime_matches(spark_version, target_runtime):
                continue

            yield RuntimeUsage(
                subscription_id=subscription_id,
                cloud=cloud_config.name.value,
                resource_group=resource_group,
                workspace_name=workspace_name,
                workspace_url=workspace_url,
                resource_type="JOB_CLUSTER",
                resource_name=job_name,
                resource_id=str(job_id),
                runtime=spark_version or "",
                detail=f"job_cluster_key={job_cluster.job_cluster_key}",
            )


def scan_workspace(
    subscription_id: str,
    workspace: Workspace,
    cloud_config: CloudConfig,
    credential: InteractiveBrowserCredential,
    target_runtime: str,
) -> list[RuntimeUsage]:
    workspace_name = workspace.name or "unknown"
    raw_workspace_url = get_workspace_url(workspace)

    if not raw_workspace_url:
        console.print(
            f"[yellow]Warning:[/] {workspace_name} has no workspace URL."
        )
        return []

    workspace_url = normalize_workspace_host(raw_workspace_url)
    resource_group = get_resource_group(workspace.id)
    console.print(f"  [cyan]{workspace_name}[/] [dim]{workspace_url}[/]")

    client = create_workspace_client(workspace, credential)
    results: list[RuntimeUsage] = []

    results.extend(
        scan_clusters(
            client=client,
            subscription_id=subscription_id,
            cloud_config=cloud_config,
            workspace_name=workspace_name,
            workspace_url=workspace_url,
            resource_group=resource_group,
            target_runtime=target_runtime,
        )
    )
    results.extend(
        scan_jobs(
            client=client,
            subscription_id=subscription_id,
            cloud_config=cloud_config,
            workspace_name=workspace_name,
            workspace_url=workspace_url,
            resource_group=resource_group,
            target_runtime=target_runtime,
        )
    )
    return results


def scan_subscription(
    subscription_id: str,
    target_runtime: str,
    cloud: AzureCloud,
) -> list[RuntimeUsage]:
    cloud_config = CLOUD_CONFIGS[cloud]
    credential = create_credential(cloud_config)

    console.rule("[bold blue]Databricks Runtime Inventory[/]")
    console.print(f"Cloud        : [cyan]{cloud.value}[/]")
    console.print(f"Subscription : {subscription_id}")
    console.print(f"Runtime      : [green]{target_runtime}[/]")

    try:
        preauthenticate(credential, cloud_config)

        console.print("[bold]Discovering Azure Databricks workspaces...[/]")
        workspaces = list_workspaces(
            subscription_id=subscription_id,
            credential=credential,
            cloud_config=cloud_config,
        )
        console.print(f"Found [bold]{len(workspaces)}[/] workspace(s).")
        console.print()

        results: list[RuntimeUsage] = []
        for index, workspace in enumerate(workspaces, start=1):
            workspace_name = workspace.name or "unknown"
            console.print(
                f"[bold][{index}/{len(workspaces)}][/] "
                f"Scanning [cyan]{workspace_name}[/]"
            )

            try:
                workspace_results = scan_workspace(
                    subscription_id=subscription_id,
                    workspace=workspace,
                    cloud_config=cloud_config,
                    credential=credential,
                    target_runtime=target_runtime,
                )
                results.extend(workspace_results)
                console.print(
                    f"  [green]Done[/] ({len(workspace_results)} matched)"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]Failed:[/] {exc}")

            console.print()

        return results
    finally:
        credential.close()


def print_results(results: list[RuntimeUsage], target_runtime: str) -> None:
    console.rule("[bold blue]Result[/]")

    if not results:
        console.print(
            f"[yellow]No resources using DBR {target_runtime} were found.[/]"
        )
        return

    table = Table(
        title=f"Databricks Runtime {target_runtime} Inventory",
        show_lines=True,
        header_style="bold blue",
    )
    table.add_column("Workspace", style="cyan", no_wrap=True)
    table.add_column("Resource Group")
    table.add_column("Type", style="magenta", no_wrap=True)
    table.add_column("Resource")
    table.add_column("Runtime", style="green", no_wrap=True)
    table.add_column("Detail")

    for item in results:
        table.add_row(
            item.workspace_name,
            item.resource_group,
            item.resource_type,
            item.resource_name,
            item.runtime,
            item.detail or "",
        )

    console.print(table)
    console.print(f"[bold green]Matched resources: {len(results)}[/]")


def write_csv(results: list[RuntimeUsage], output_path: Path) -> Path:
    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(RuntimeUsage)]

    with resolved_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)

    return resolved_path


@app.command()
def scan(
    subscription_id: Annotated[
        str,
        typer.Option(
            "--subscription",
            "-s",
            help="Azure subscription ID.",
        ),
    ],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            "-r",
            help="Target Databricks Runtime, for example 15.4.",
        ),
    ],
    cloud: Annotated[
        AzureCloud,
        typer.Option(
            "--cloud",
            "-c",
            help="Azure cloud environment: global or china.",
            case_sensitive=False,
        ),
    ] = AzureCloud.GLOBAL,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional CSV output path.",
        ),
    ] = None,
) -> None:
    """Scan all Azure Databricks workspaces in a subscription."""
    subscription_id = subscription_id.strip()
    runtime = runtime.strip()

    if not subscription_id:
        raise typer.BadParameter("Subscription ID cannot be empty.")
    if not runtime:
        raise typer.BadParameter("Runtime cannot be empty.")

    try:
        results = scan_subscription(
            subscription_id=subscription_id,
            target_runtime=runtime,
            cloud=cloud,
        )
        print_results(results, runtime)

        if output is not None:
            csv_path = write_csv(results, output)
            console.print(f"[bold green]CSV saved:[/] {csv_path}")
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Scan cancelled.[/]")
        raise typer.Exit(code=130) from None
    except Exception as exc:
        console.print()
        console.print(f"[bold red]Scan failed:[/] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
