from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import typer
from azure.core import AzureClouds
from azure.identity import AzureAuthorityHosts, DefaultAzureCredential
from azure.mgmt.databricks import AzureDatabricksManagementClient
from azure.mgmt.databricks.models import Workspace
from databricks.sdk import WorkspaceClient
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="dbr-inventory",
    help="Scan Azure Databricks resources for a specific DBR runtime.",
    no_args_is_help=True,
)
console = Console()


class AzureCloud(str, Enum):
    GLOBAL = "global"
    CHINA = "china"


@dataclass(frozen=True)
class CloudConfig:
    name: AzureCloud
    authority: str
    arm_cloud: AzureClouds
    databricks_environment: str


CLOUD_CONFIGS: dict[AzureCloud, CloudConfig] = {
    AzureCloud.GLOBAL: CloudConfig(
        name=AzureCloud.GLOBAL,
        authority=AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
        arm_cloud=AzureClouds.AZURE_PUBLIC_CLOUD,
        databricks_environment="PUBLIC",
    ),
    AzureCloud.CHINA: CloudConfig(
        name=AzureCloud.CHINA,
        authority=AzureAuthorityHosts.AZURE_CHINA,
        arm_cloud=AzureClouds.AZURE_CHINA_CLOUD,
        databricks_environment="CHINA",
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


def create_credential(cloud_config: CloudConfig) -> DefaultAzureCredential:
    return DefaultAzureCredential(authority=cloud_config.authority)


def runtime_matches(spark_version: str | None, target_runtime: str) -> bool:
    if not spark_version:
        return False
    normalized_target = target_runtime.strip().rstrip(".")
    return (
        spark_version == normalized_target  # noqa: PIE810
        or spark_version.startswith(f"{normalized_target}.")
        or spark_version.startswith(f"{normalized_target}-")
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
    if workspace_url.startswith("https://"):
        return workspace_url.rstrip("/")
    return f"https://{workspace_url.rstrip('/')}"


def list_workspaces(
    subscription_id: str,
    credential: DefaultAzureCredential,
    cloud_config: CloudConfig,
) -> list[Workspace]:
    client = AzureDatabricksManagementClient(
        credential=credential,
        subscription_id=subscription_id,
        cloud_setting=cloud_config.arm_cloud,
    )
    return list(client.workspaces.list_by_subscription())


def create_workspace_client(
    workspace: Workspace,
    cloud_config: CloudConfig,
) -> WorkspaceClient:
    properties = workspace.properties
    if properties is None:
        raise RuntimeError("Workspace does not contain properties.")
    workspace_url = properties.workspace_url
    if not workspace_url:
        raise RuntimeError("Workspace does not contain workspace_url.")
    if not workspace.id:
        raise RuntimeError("Workspace does not contain Azure resource ID.")

    return WorkspaceClient(
        host=normalize_workspace_host(workspace_url),
        azure_workspace_resource_id=workspace.id,
        azure_environment=cloud_config.databricks_environment,
        auth_type="azure-cli",
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

        cluster_name = cluster.cluster_name or cluster.cluster_id or "unknown"
        detail = cluster.state.value if cluster.state is not None else None

        yield RuntimeUsage(
            subscription_id=subscription_id,
            cloud=cloud_config.name.value,
            resource_group=resource_group,
            workspace_name=workspace_name,
            workspace_url=workspace_url,
            resource_type="CLUSTER",
            resource_name=cluster_name,
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
            console.print(f"[yellow]Warning:[/] Unable to read job {job_id}: {exc}")
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
    target_runtime: str,
) -> list[RuntimeUsage]:
    workspace_name = workspace.name or "unknown"
    properties = workspace.properties
    if properties is None:
        console.print(f"[yellow]Warning:[/] {workspace_name} has no properties.")
        return []

    workspace_url = properties.workspace_url
    if not workspace_url:
        console.print(f"[yellow]Warning:[/] {workspace_name} has no workspace URL.")
        return []

    workspace_url = normalize_workspace_host(workspace_url)
    resource_group = get_resource_group(workspace.id)
    console.print(f"  [cyan]{workspace_name}[/] [dim]{workspace_url}[/]")

    client = create_workspace_client(workspace=workspace, cloud_config=cloud_config)
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

    console.print()
    console.rule("[bold blue]Databricks Runtime Inventory[/]")
    console.print(f"Cloud        : [cyan]{cloud.value}[/]")
    console.print(f"Subscription : [white]{subscription_id}[/]")
    console.print(f"Runtime      : [green]{target_runtime}[/]")
    console.print("\n[bold]Discovering Azure Databricks workspaces...[/]")

    workspaces = list_workspaces(
        subscription_id=subscription_id,
        credential=credential,
        cloud_config=cloud_config,
    )
    console.print(f"Found [bold]{len(workspaces)}[/] workspace(s).\n")

    results: list[RuntimeUsage] = []
    for index, workspace in enumerate(workspaces, start=1):
        workspace_name = workspace.name or "unknown"
        console.print(
            f"[bold][{index}/{len(workspaces)}][/bold] "
            f"Scanning [cyan]{workspace_name}[/]"
        )
        try:
            workspace_results = scan_workspace(
                subscription_id=subscription_id,
                workspace=workspace,
                cloud_config=cloud_config,
                target_runtime=target_runtime,
            )
            results.extend(workspace_results)
            console.print(f"  [green]Done[/] ({len(workspace_results)} matched)")
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Failed:[/] {exc}")
        console.print()

    return results


def print_results(results: list[RuntimeUsage], target_runtime: str) -> None:
    console.rule("[bold blue]Result[/]")
    if not results:
        console.print(f"\n[yellow]No resources using DBR {target_runtime} were found.[/]")
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

    console.print()
    console.print(table)
    console.print(f"\n[bold green]Matched resources: {len(results)}[/]")


@app.command()
def scan(
    subscription_id: Annotated[
        str,
        typer.Option("--subscription", "-s", help="Azure subscription ID."),
    ],
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            "-r",
            help="Target Databricks Runtime version, e.g. 15.4.",
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
) -> None:
    """Scan an Azure subscription for Databricks resources using a DBR version."""
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
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan cancelled.[/]")
        raise typer.Exit(code=130) from None
    except Exception as exc:
        console.print(f"\n[bold red]Scan failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    print_results(results=results, target_runtime=runtime)


def main() -> None:
    app()

if __name__ == "__main__":
    main()
    
