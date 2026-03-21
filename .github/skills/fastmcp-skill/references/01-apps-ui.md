# Apps and UI

Source lines: 1-1045 from the original FastMCP documentation dump.

Custom HTML apps, prefab apps, UI patterns, charts, forms, status displays, and app wiring.

---

# Custom HTML Apps
Source: https://gofastmcp.com/apps/low-level

Build apps with your own HTML, CSS, and JavaScript using the MCP Apps extension directly.

<VersionBadge />

The [MCP Apps extension](https://modelcontextprotocol.io/docs/extensions/apps) is an open protocol that lets tools return interactive UIs — an HTML page rendered in a sandboxed iframe inside the host client. [Prefab UI](/apps/prefab) builds on this protocol so you never have to think about it, but when you need full control — custom rendering, a specific JavaScript framework, maps, 3D, video — you can use the MCP Apps extension directly.

This page covers how to write custom HTML apps and wire them up in FastMCP. You'll be working with the [`@modelcontextprotocol/ext-apps`](https://github.com/modelcontextprotocol/ext-apps) JavaScript SDK for host communication, and FastMCP's `AppConfig` for resource and CSP management.

## How It Works

An MCP App has two parts:

1. A **tool** that does the work and returns data
2. A **`ui://` resource** containing the HTML that renders that data

The tool declares which resource to use via `AppConfig`. When the host calls the tool, it also fetches the linked resource, renders it in a sandboxed iframe, and pushes the tool result into the app via `postMessage`. The app can also call tools back, enabling interactive workflows.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import json

from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP

mcp = FastMCP("My App Server")

# The tool does the work
@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
def generate_chart(data: list[float]) -> str:
    return json.dumps({"values": data})

# The resource provides the UI
@mcp.resource("ui://my-app/view.html")
def chart_view() -> str:
    return "<html>...</html>"
```

## AppConfig

`AppConfig` controls how a tool or resource participates in the Apps extension. Import it from `fastmcp.server.apps`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.apps import AppConfig
```

On **tools**, you'll typically set `resource_uri` to point to the UI resource:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
def my_tool() -> str:
    return "result"
```

You can also pass a raw dict with camelCase keys, matching the wire format:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool(app={"resourceUri": "ui://my-app/view.html"})
def my_tool() -> str:
    return "result"
```

### Tool Visibility

The `visibility` field controls where a tool appears:

* `["model"]` — visible to the LLM (the default behavior)
* `["app"]` — only callable from within the app UI, hidden from the LLM
* `["model", "app"]` — both

This is useful when you have tools that only make sense as part of the app's interactive flow, not as standalone LLM actions.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool(
    app=AppConfig(
        resource_uri="ui://my-app/view.html",
        visibility=["app"],
    )
)
def refresh_data() -> str:
    """Only callable from the app UI, not by the LLM."""
    return fetch_latest()
```

### AppConfig Fields

| Field            | Type                  | Description                                                      |
| ---------------- | --------------------- | ---------------------------------------------------------------- |
| `resource_uri`   | `str`                 | URI of the UI resource. Tools only.                              |
| `visibility`     | `list[str]`           | Where the tool appears: `"model"`, `"app"`, or both. Tools only. |
| `csp`            | `ResourceCSP`         | Content Security Policy for the iframe.                          |
| `permissions`    | `ResourcePermissions` | Iframe sandbox permissions.                                      |
| `domain`         | `str`                 | Stable sandbox origin for the iframe.                            |
| `prefers_border` | `bool`                | Whether the UI prefers a visible border.                         |

<Note>
  On **resources**, `resource_uri` and `visibility` must not be set — the resource *is* the UI. Use `AppConfig` on resources only for `csp`, `permissions`, and other display settings.
</Note>

## UI Resources

Resources using the `ui://` scheme are automatically served with the MIME type `text/html;profile=mcp-app`. You don't need to set this manually.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.resource("ui://my-app/view.html")
def my_view() -> str:
    return "<html>...</html>"
```

The HTML can be anything — a full single-page app, a simple display, or a complex interactive tool. The host renders it in a sandboxed iframe and establishes a `postMessage` channel for communication.

### Writing the App HTML

Your HTML app communicates with the host using the [`@modelcontextprotocol/ext-apps`](https://github.com/modelcontextprotocol/ext-apps) JavaScript SDK. The simplest approach is to load it from a CDN:

```html theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
<script type="module">
  import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@0.4.0/app-with-deps";

  const app = new App({ name: "My App", version: "1.0.0" });

  // Receive tool results pushed by the host
  app.ontoolresult = ({ content }) => {
    const text = content?.find(c => c.type === 'text');
    if (text) {
      document.getElementById('output').textContent = text.text;
    }
  };

  // Connect to the host
  await app.connect();
</script>
```

The `App` object provides:

* **`app.ontoolresult`** — callback that receives tool results pushed by the host
* **`app.callServerTool({name, arguments})`** — call a tool on the server from within the app
* **`app.onhostcontextchanged`** — callback for host context changes (e.g., safe area insets)
* **`app.getHostContext()`** — get current host context

<Note>
  If your HTML loads external scripts, styles, or makes API calls, you need to declare those domains in the CSP configuration. See [Security](#security) below.
</Note>

## Security

Apps run in sandboxed iframes with a deny-by-default Content Security Policy. By default, only inline scripts and styles are allowed — no external network access.

### Content Security Policy

If your app needs to load external resources (CDN scripts, API calls, embedded iframes), declare the allowed domains with `ResourceCSP`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.apps import AppConfig, ResourceCSP

@mcp.resource(
    "ui://my-app/view.html",
    app=AppConfig(
        csp=ResourceCSP(
            resource_domains=["https://unpkg.com", "https://cdn.example.com"],
            connect_domains=["https://api.example.com"],
        )
    ),
)
def my_view() -> str:
    return "<html>...</html>"
```

| CSP Field          | Controls                                            |
| ------------------ | --------------------------------------------------- |
| `connect_domains`  | `fetch`, XHR, WebSocket (`connect-src`)             |
| `resource_domains` | Scripts, images, styles, fonts (`script-src`, etc.) |
| `frame_domains`    | Nested iframes (`frame-src`)                        |
| `base_uri_domains` | Document base URI (`base-uri`)                      |

### Permissions

If your app needs browser capabilities like camera or clipboard access, request them via `ResourcePermissions`:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.apps import AppConfig, ResourcePermissions

@mcp.resource(
    "ui://my-app/view.html",
    app=AppConfig(
        permissions=ResourcePermissions(
            camera={},
            clipboard_write={},
        )
    ),
)
def my_view() -> str:
    return "<html>...</html>"
```

Hosts may or may not grant these permissions. Your app should use JavaScript feature detection as a fallback.

## Example: QR Code Server

This example creates a tool that generates QR codes and an app that renders them as images. It's based on the [official MCP Apps example](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/qr-server). Requires the `qrcode[pil]` package.

```python expandable theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
import base64
import io

import qrcode
from mcp import types

from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP
from fastmcp.tools import ToolResult

mcp = FastMCP("QR Code Server")

VIEW_URI = "ui://qr-server/view.html"


@mcp.tool(app=AppConfig(resource_uri=VIEW_URI))
def generate_qr(text: str = "https://gofastmcp.com") -> ToolResult:
    """Generate a QR code from text."""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()

    return ToolResult(
        content=[types.ImageContent(type="image", data=b64, mimeType="image/png")]
    )


@mcp.resource(
    VIEW_URI,
    app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
)
def view() -> str:
    """Interactive QR code viewer."""
    return """\
<!DOCTYPE html>
<html>
<head>
  <meta name="color-scheme" content="light dark">
  <style>
    body { display: flex; justify-content: center;
           align-items: center; height: 340px; width: 340px;
           margin: 0; background: transparent; }
    img  { width: 300px; height: 300px; border-radius: 8px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  </style>
</head>
<body>
  <div id="qr"></div>
  <script type="module">
    import { App } from
      "https://unpkg.com/@modelcontextprotocol/ext-apps@0.4.0/app-with-deps";

    const app = new App({ name: "QR View", version: "1.0.0" });

    app.ontoolresult = ({ content }) => {
      const img = content?.find(c => c.type === 'image');
      if (img) {
        const el = document.createElement('img');
        el.src = `data:${img.mimeType};base64,${img.data}`;
        el.alt = "QR Code";
        document.getElementById('qr').replaceChildren(el);
      }
    };

    await app.connect();
  </script>
</body>
</html>"""
```

The tool generates a QR code as a base64 PNG. The resource loads the MCP Apps JS SDK from unpkg (declared in the CSP), listens for tool results, and renders the image. The host wires them together — when the LLM calls `generate_qr`, the QR code appears in an interactive frame inside the conversation.

## Checking Client Support

Not all hosts support the Apps extension. You can check at runtime using the tool's [context](/servers/context):

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import Context
from fastmcp.server.apps import AppConfig, UI_EXTENSION_ID

@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
async def my_tool(ctx: Context) -> str:
    if ctx.client_supports_extension(UI_EXTENSION_ID):
        # Return data optimized for UI rendering
        return rich_response()
    else:
        # Fall back to plain text
        return plain_text_response()
```


# Apps
Source: https://gofastmcp.com/apps/overview

Give your tools interactive UIs rendered directly in the conversation.

<VersionBadge />

MCP Apps let your tools return interactive UIs — rendered in a sandboxed iframe right inside the host client's conversation. Instead of returning plain text, a tool can show a chart, a sortable table, a form, or anything you can build with HTML.

FastMCP implements the [MCP Apps extension](https://modelcontextprotocol.io/docs/extensions/apps) and provides two approaches:

## Prefab Apps (Recommended)

<VersionBadge />

<Tip>
  [Prefab](https://prefab.prefect.io) is in extremely early, active development — its API changes frequently and breaking changes can occur with any release. The FastMCP integration is equally new and under rapid development. These docs are included for users who want to work on the cutting edge; production use is not recommended. Always [pin `prefab-ui` to a specific version](/apps/prefab#getting-started) in your dependencies.
</Tip>

[Prefab UI](https://prefab.prefect.io) is a declarative UI framework for Python. You describe layouts, charts, tables, forms, and interactive behaviors using a Python DSL — and the framework compiles them to a JSON protocol that a shared renderer interprets. It started as a component library inside FastMCP and grew into its own framework with [comprehensive documentation](https://prefab.prefect.io).

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, BarChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Dashboard")

@mcp.tool(app=True)
def sales_chart(year: int) -> PrefabApp:
    """Show sales data as an interactive chart."""
    data = get_sales_data(year)

    with Column(gap=4, css_class="p-6") as view:
        Heading(f"{year} Sales")
        BarChart(
            data=data,
            series=[ChartSeries(data_key="revenue", label="Revenue")],
            x_axis="month",
        )

    return PrefabApp(view=view)
```

Install with `pip install "fastmcp[apps]"` and see [Prefab Apps](/apps/prefab) for the integration guide.

## Custom HTML Apps

The [MCP Apps extension](https://modelcontextprotocol.io/docs/extensions/apps) is an open protocol, and you can use it directly when you need full control. You write your own HTML/CSS/JavaScript and communicate with the host via the [`@modelcontextprotocol/ext-apps`](https://github.com/modelcontextprotocol/ext-apps) SDK.

This is the right choice for custom rendering (maps, 3D, video), specific JavaScript frameworks, or capabilities beyond what the component library offers.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp import FastMCP
from fastmcp.server.apps import AppConfig, ResourceCSP

mcp = FastMCP("Custom App")

@mcp.tool(app=AppConfig(resource_uri="ui://my-app/view.html"))
def my_tool() -> str:
    return '{"values": [1, 2, 3]}'

@mcp.resource(
    "ui://my-app/view.html",
    app=AppConfig(csp=ResourceCSP(resource_domains=["https://unpkg.com"])),
)
def view() -> str:
    return "<html>...</html>"
```

See [Custom HTML Apps](/apps/low-level) for the full reference.


# Patterns
Source: https://gofastmcp.com/apps/patterns

Charts, tables, forms, and other common tool UIs.

<VersionBadge />

<Tip>
  [Prefab](https://prefab.prefect.io) is in extremely early, active development — its API changes frequently and breaking changes can occur with any release. The FastMCP integration is equally new and under rapid development. These docs are included for users who want to work on the cutting edge; production use is not recommended. Always pin `prefab-ui` to a specific version in your dependencies.
</Tip>

The most common use of Prefab is giving your tools a visual representation — a chart instead of raw numbers, a sortable table instead of a text dump, a status dashboard instead of a list of booleans. Each pattern below is a complete, copy-pasteable tool.

## Charts

Prefab includes [bar, line, area, pie, radar, and radial charts](https://prefab.prefect.io/docs/components/charts). They all render client-side with tooltips, legends, and responsive sizing.

### Bar Chart

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, BarChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Charts")


@mcp.tool(app=True)
def quarterly_revenue(year: int) -> PrefabApp:
    """Show quarterly revenue as a bar chart."""
    data = [
        {"quarter": "Q1", "revenue": 42000, "costs": 28000},
        {"quarter": "Q2", "revenue": 51000, "costs": 31000},
        {"quarter": "Q3", "revenue": 47000, "costs": 29000},
        {"quarter": "Q4", "revenue": 63000, "costs": 35000},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading(f"{year} Revenue vs Costs")
        BarChart(
            data=data,
            series=[
                ChartSeries(data_key="revenue", label="Revenue"),
                ChartSeries(data_key="costs", label="Costs"),
            ],
            x_axis="quarter",
            show_legend=True,
        )

    return PrefabApp(view=view)
```

Multiple `ChartSeries` entries plot different data keys. Add `stacked=True` to stack bars, or `horizontal=True` to flip the axes.

### Area Chart

`LineChart` and `AreaChart` share the same API as `BarChart`, with `curve` for interpolation (`"linear"`, `"smooth"`, `"step"`) and `show_dots` for data points:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, AreaChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Charts")


@mcp.tool(app=True)
def usage_trend() -> PrefabApp:
    """Show API usage over time."""
    data = [
        {"date": "Feb 1", "requests": 1200},
        {"date": "Feb 2", "requests": 1350},
        {"date": "Feb 3", "requests": 980},
        {"date": "Feb 4", "requests": 1500},
        {"date": "Feb 5", "requests": 1420},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading("API Usage")
        AreaChart(
            data=data,
            series=[ChartSeries(data_key="requests", label="Requests")],
            x_axis="date",
            curve="smooth",
            height=250,
        )

    return PrefabApp(view=view)
```

### Pie and Donut Charts

`PieChart` uses `data_key` (the numeric value) and `name_key` (the label) instead of series. Set `inner_radius` for a donut:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, PieChart
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Charts")


@mcp.tool(app=True)
def ticket_breakdown() -> PrefabApp:
    """Show open tickets by category."""
    data = [
        {"category": "Bug", "count": 23},
        {"category": "Feature", "count": 15},
        {"category": "Docs", "count": 8},
        {"category": "Infra", "count": 12},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading("Open Tickets")
        PieChart(
            data=data,
            data_key="count",
            name_key="category",
            show_legend=True,
            inner_radius=60,
        )

    return PrefabApp(view=view)
```

## Data Tables

[DataTable](https://prefab.prefect.io/docs/components/data-display/data-table) provides sortable columns, full-text search, and pagination — all running client-side in the browser.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, DataTable, DataTableColumn
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Directory")


@mcp.tool(app=True)
def employee_directory() -> PrefabApp:
    """Show a searchable, sortable employee directory."""
    employees = [
        {"name": "Alice Chen", "department": "Engineering", "role": "Staff Engineer", "location": "SF"},
        {"name": "Bob Martinez", "department": "Design", "role": "Lead Designer", "location": "NYC"},
        {"name": "Carol Johnson", "department": "Engineering", "role": "Senior Engineer", "location": "London"},
        {"name": "David Kim", "department": "Product", "role": "Product Manager", "location": "SF"},
        {"name": "Eva Müller", "department": "Engineering", "role": "Engineer", "location": "Berlin"},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading("Employee Directory")
        DataTable(
            columns=[
                DataTableColumn(key="name", header="Name", sortable=True),
                DataTableColumn(key="department", header="Department", sortable=True),
                DataTableColumn(key="role", header="Role"),
                DataTableColumn(key="location", header="Office", sortable=True),
            ],
            rows=employees,
            searchable=True,
            paginated=True,
            page_size=15,
        )

    return PrefabApp(view=view)
```

## Forms

A form collects input, but it needs somewhere to send that input. The [`CallTool`](https://prefab.prefect.io/docs/concepts/actions) action connects a form to a tool on your MCP server — so you need two tools: one that renders the form, and one that handles the submission.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import (
    Column, Heading, Row, Muted, Badge, Input, Select,
    Textarea, Button, Form, ForEach, Separator,
)
from prefab_ui.actions import ShowToast
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Contacts")

contacts_db: list[dict] = [
    {"name": "Zaphod Beeblebrox", "email": "zaphod@galaxy.gov", "category": "Partner"},
]


@mcp.tool(app=True)
def contact_form() -> PrefabApp:
    """Show a contact list with a form to add new contacts."""
    with Column(gap=6, css_class="p-6") as view:
        Heading("Contacts")

        with ForEach("contacts"):
            with Row(gap=2, align="center"):
                Muted("{{ name }}")
                Muted("{{ email }}")
                Badge("{{ category }}")

        Separator()

        with Form(
            on_submit=CallTool(
                "save_contact",
                result_key="contacts",
                on_success=ShowToast("Contact saved!", variant="success"),
                on_error=ShowToast("{{ $error }}", variant="error"),
            )
        ):
            Input(name="name", label="Full Name", required=True)
            Input(name="email", label="Email", input_type="email", required=True)
            Select(
                name="category",
                label="Category",
                options=["Customer", "Vendor", "Partner", "Other"],
            )
            Textarea(name="notes", label="Notes", placeholder="Optional notes...")
            Button("Save Contact")

    return PrefabApp(view=view, state={"contacts": list(contacts_db)})


@mcp.tool
def save_contact(
    name: str,
    email: str,
    category: str = "Other",
    notes: str = "",
) -> list[dict]:
    """Save a new contact and return the updated list."""
    contacts_db.append({"name": name, "email": email, "category": category, "notes": notes})
    return list(contacts_db)
```

When the user submits the form, the renderer calls `save_contact` on the server with all named input values as arguments. Because `result_key="contacts"` is set, the returned list replaces the `contacts` state — and the `ForEach` re-renders with the new data automatically.

The `save_contact` tool is a regular MCP tool. The LLM can also call it directly in conversation. Your UI actions and your conversational tools are the same thing.

### Pydantic Model Forms

For complex forms, `Form.from_model()` generates the entire form from a Pydantic model — inputs, labels, validation, and submit wiring:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from typing import Literal

from pydantic import BaseModel, Field
from prefab_ui.components import Column, Heading, Form
from prefab_ui.actions.mcp import CallTool
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Bug Tracker")


class BugReport(BaseModel):
    title: str = Field(title="Bug Title")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        title="Severity", default="medium"
    )
    description: str = Field(title="Description")
    steps_to_reproduce: str = Field(title="Steps to Reproduce")


@mcp.tool(app=True)
def report_bug() -> PrefabApp:
    """Show a bug report form."""
    with Column(gap=4, css_class="p-6") as view:
        Heading("Report a Bug")
        Form.from_model(BugReport, on_submit=CallTool("create_bug_report"))

    return PrefabApp(view=view)


@mcp.tool
def create_bug_report(data: dict) -> str:
    """Create a bug report from the form submission."""
    report = BugReport(**data)
    # save to database...
    return f"Created bug report: {report.title}"
```

`str` fields become text inputs, `Literal` becomes a select, `bool` becomes a checkbox. The `on_submit` CallTool receives all field values under a `data` key.

## Status Displays

Cards, badges, progress bars, and grids combine naturally for dashboards. See the [Prefab layout](https://prefab.prefect.io/docs/concepts/composition) and [container](https://prefab.prefect.io/docs/components/containers) docs for the full set of layout and display components.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import (
    Column, Row, Grid, Heading, Text, Muted, Badge,
    Card, CardContent, Progress, Separator,
)
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Monitoring")


@mcp.tool(app=True)
def system_status() -> PrefabApp:
    """Show current system health."""
    services = [
        {"name": "API Gateway", "status": "healthy", "ok": True, "latency_ms": 12, "uptime_pct": 99.9},
        {"name": "Database", "status": "healthy", "ok": True, "latency_ms": 3, "uptime_pct": 99.99},
        {"name": "Cache", "status": "degraded", "ok": False, "latency_ms": 45, "uptime_pct": 98.2},
        {"name": "Queue", "status": "healthy", "ok": True, "latency_ms": 8, "uptime_pct": 99.8},
    ]
    all_ok = all(s["ok"] for s in services)

    with Column(gap=4, css_class="p-6") as view:
        with Row(gap=2, align="center"):
            Heading("System Status")
            Badge(
                "All Healthy" if all_ok else "Degraded",
                variant="success" if all_ok else "destructive",
            )

        Separator()

        with Grid(columns=2, gap=4):
            for svc in services:
                with Card():
                    with CardContent():
                        with Row(gap=2, align="center"):
                            Text(svc["name"], css_class="font-medium")
                            Badge(
                                svc["status"],
                                variant="success" if svc["ok"] else "destructive",
                            )
                        Muted(f"Response: {svc['latency_ms']}ms")
                        Progress(value=svc["uptime_pct"])

    return PrefabApp(view=view)
```

## Conditional Content

[`If`, `Elif`, and `Else`](https://prefab.prefect.io/docs/concepts/composition#conditional-rendering) show or hide content based on state. Changes are instant — no server round-trip.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, Switch, Separator, Alert, If
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Flags")


@mcp.tool(app=True)
def feature_flags() -> PrefabApp:
    """Toggle feature flags with live preview."""
    with Column(gap=4, css_class="p-6") as view:
        Heading("Feature Flags")

        Switch(name="dark_mode", label="Dark Mode")
        Switch(name="beta_features", label="Beta Features")

        Separator()

        with If("{{ dark_mode }}"):
            Alert(title="Dark mode enabled", description="UI will use dark theme.")
        with If("{{ beta_features }}"):
            Alert(
                title="Beta features active",
                description="Experimental features are now visible.",
                variant="warning",
            )

    return PrefabApp(view=view, state={"dark_mode": False, "beta_features": False})
```

## Tabs

[Tabs](https://prefab.prefect.io/docs/components/containers/tabs) organize content into switchable views. Switching is client-side — no server round-trip.

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import (
    Column, Heading, Text, Muted, Badge, Row,
    DataTable, DataTableColumn, Tabs, Tab, ForEach,
)
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Projects")


@mcp.tool(app=True)
def project_overview(project_id: str) -> PrefabApp:
    """Show project details organized in tabs."""
    project = {
        "name": "FastMCP v3",
        "description": "Next generation MCP framework with Apps support.",
        "status": "Active",
        "created_at": "2025-01-15",
        "members": [
            {"name": "Alice Chen", "role": "Lead"},
            {"name": "Bob Martinez", "role": "Design"},
        ],
        "activity": [
            {"timestamp": "2 hours ago", "message": "Merged PR #342"},
            {"timestamp": "1 day ago", "message": "Released v3.0.1"},
        ],
    }

    with Column(gap=4, css_class="p-6") as view:
        Heading(project["name"])

        with Tabs():
            with Tab("Overview"):
                Text(project["description"])
                with Row(gap=4):
                    Badge(project["status"])
                    Muted(f"Created: {project['created_at']}")

            with Tab("Members"):
                DataTable(
                    columns=[
                        DataTableColumn(key="name", header="Name", sortable=True),
                        DataTableColumn(key="role", header="Role"),
                    ],
                    rows=project["members"],
                )

            with Tab("Activity"):
                with ForEach("activity"):
                    with Row(gap=2):
                        Muted("{{ timestamp }}")
                        Text("{{ message }}")

    return PrefabApp(view=view, state={"activity": project["activity"]})
```

## Accordion

[Accordion](https://prefab.prefect.io/docs/components/containers/accordion) collapses sections to save space. `multiple=True` lets users expand several items at once:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import (
    Column, Heading, Row, Text, Badge, Progress,
    Accordion, AccordionItem,
)
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("API Monitor")


@mcp.tool(app=True)
def api_health() -> PrefabApp:
    """Show health details for each API endpoint."""
    endpoints = [
        {"path": "/api/users", "status": 200, "healthy": True, "avg_ms": 45, "p99_ms": 120, "uptime_pct": 99.9},
        {"path": "/api/orders", "status": 200, "healthy": True, "avg_ms": 82, "p99_ms": 250, "uptime_pct": 99.7},
        {"path": "/api/search", "status": 200, "healthy": True, "avg_ms": 150, "p99_ms": 500, "uptime_pct": 99.5},
        {"path": "/api/webhooks", "status": 503, "healthy": False, "avg_ms": 2000, "p99_ms": 5000, "uptime_pct": 95.1},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading("API Health")

        with Accordion(multiple=True):
            for ep in endpoints:
                with AccordionItem(ep["path"]):
                    with Row(gap=4):
                        Badge(
                            f"{ep['status']}",
                            variant="success" if ep["healthy"] else "destructive",
                        )
                        Text(f"Avg: {ep['avg_ms']}ms")
                        Text(f"P99: {ep['p99_ms']}ms")
                    Progress(value=ep["uptime_pct"])

    return PrefabApp(view=view)
```

## Next Steps

* **[Custom HTML Apps](/apps/low-level)** — When you need your own HTML, CSS, and JavaScript
* **[Prefab UI Docs](https://prefab.prefect.io)** — Components, state, expressions, and actions


# Prefab Apps
Source: https://gofastmcp.com/apps/prefab

Build interactive tool UIs in pure Python — no HTML or JavaScript required.

<VersionBadge />

<Tip>
  [Prefab](https://prefab.prefect.io) is in extremely early, active development — its API changes frequently and breaking changes can occur with any release. The FastMCP integration is equally new and under rapid development. These docs are included for users who want to work on the cutting edge; production use is not recommended. Always pin `prefab-ui` to a specific version in your dependencies (see below).
</Tip>

[Prefab UI](https://prefab.prefect.io) is a declarative UI framework for Python. You describe what your interface should look like — a chart, a table, a form — and return it from your tool. FastMCP takes care of everything else: registering the renderer, wiring the protocol metadata, and delivering the component tree to the host.

Prefab started as a component library inside FastMCP and grew into a full framework for building interactive applications — with its own state management, reactive expression system, and action model. The [Prefab documentation](https://prefab.prefect.io) covers all of this in depth. This page focuses on the FastMCP integration: what you return from a tool, and what FastMCP does with it.

```bash theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
pip install "fastmcp[apps]"
```

<Tip>
  Prefab UI is in active early development and its API changes frequently. We strongly recommend pinning `prefab-ui` to a specific version in your project's dependencies. Installing `fastmcp[apps]` pulls in `prefab-ui` but won't pin it — so a routine `pip install --upgrade` could introduce breaking changes.

  ```toml theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
  # pyproject.toml
  dependencies = [
      "fastmcp[apps]",
      "prefab-ui==0.8.0",  # pin to a known working version
  ]
  ```
</Tip>

Here's the simplest possible Prefab App — a tool that returns a bar chart:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, BarChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Dashboard")


@mcp.tool(app=True)
def revenue_chart(year: int) -> PrefabApp:
    """Show annual revenue as an interactive bar chart."""
    data = [
        {"quarter": "Q1", "revenue": 42000},
        {"quarter": "Q2", "revenue": 51000},
        {"quarter": "Q3", "revenue": 47000},
        {"quarter": "Q4", "revenue": 63000},
    ]

    with Column(gap=4, css_class="p-6") as view:
        Heading(f"{year} Revenue")
        BarChart(
            data=data,
            series=[ChartSeries(data_key="revenue", label="Revenue")],
            x_axis="quarter",
        )

    return PrefabApp(view=view)
```

That's it — you declare a layout using Python's `with` statement, and return it. When the host calls this tool, the user sees an interactive bar chart instead of a JSON blob. The [Patterns](/apps/patterns) page has more examples: area charts, data tables, forms, status dashboards, and more.

## What You Return

### Components

The simplest way to get started. If you're returning a visual representation of data and don't need Prefab's more advanced features like initial state or stylesheets, just return the components directly. FastMCP wraps them in a `PrefabApp` automatically:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, Badge
from fastmcp import FastMCP

mcp = FastMCP("Status")


@mcp.tool(app=True)
def status_badge() -> Column:
    """Show system status."""
    with Column(gap=2) as view:
        Heading("All Systems Operational")
        Badge("Healthy", variant="success")
    return view
```

Want a chart? Return a chart. Want a table? Return a table. FastMCP handles the wiring.

### PrefabApp

When you need more control — setting initial state values that components can read and react to, or configuring the rendering engine — return a `PrefabApp` explicitly:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, Text, Button, If, Badge
from prefab_ui.actions import ToggleState
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool(app=True)
def toggle_demo() -> PrefabApp:
    """Interactive toggle with state."""
    with Column(gap=4, css_class="p-6") as view:
        Button("Toggle", on_click=ToggleState("show"))
        with If("{{ show }}"):
            Badge("Visible!", variant="success")

    return PrefabApp(view=view, state={"show": False})
```

The `state` dict provides the initial values. Components reference state with `{{ expression }}` templates. State mutations like `ToggleState` happen entirely in the browser — no server round-trip. The [Prefab state guide](https://prefab.prefect.io/docs/concepts/state) covers this in detail.

### ToolResult

Every tool result has two audiences: the renderer (which displays the UI) and the LLM (which reads the text content to understand what happened). By default, Prefab Apps send `"[Rendered Prefab UI]"` as the text content, which tells the LLM almost nothing.

If you want the LLM to understand the result — so it can reference the data in conversation, summarize it, or decide what to do next — wrap your return in a `ToolResult` with a meaningful `content` string:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from prefab_ui.components import Column, Heading, BarChart, ChartSeries
from prefab_ui.app import PrefabApp
from fastmcp import FastMCP
from fastmcp.tools import ToolResult

mcp = FastMCP("Sales")


@mcp.tool(app=True)
def sales_overview(year: int) -> ToolResult:
    """Show sales data visually and summarize for the model."""
    data = get_sales_data(year)
    total = sum(row["revenue"] for row in data)

    with Column(gap=4, css_class="p-6") as view:
        Heading("Sales Overview")
        BarChart(data=data, series=[ChartSeries(data_key="revenue")])

    return ToolResult(
        content=f"Total revenue for {year}: ${total:,} across {len(data)} quarters",
        structured_content=view,
    )
```

The user sees the chart. The LLM sees `"Total revenue for 2025: $203,000 across 4 quarters"` and can reason about it.

## Type Inference

If your tool's return type annotation is a Prefab type — `PrefabApp`, `Component`, or their `Optional` variants — FastMCP detects this and enables app rendering automatically:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
@mcp.tool
def greet(name: str) -> PrefabApp:
    return PrefabApp(view=Heading(f"Hello, {name}!"))
```

This is equivalent to `@mcp.tool(app=True)`. Explicit `app=True` is recommended for clarity, and is required when the return type doesn't reveal a Prefab type (e.g., `-> ToolResult`).

## How It Works

Behind the scenes, when a tool returns a Prefab component or `PrefabApp`, FastMCP:

1. **Registers a shared renderer** — a `ui://prefab/renderer.html` resource containing the JavaScript rendering engine, fetched once by the host and reused across all your Prefab tools.
2. **Wires the tool metadata** — so the host knows to load the renderer iframe when displaying the tool result.
3. **Serializes the component tree** — your Python components become `structuredContent` on the tool result, which the renderer interprets and displays.

None of this requires any configuration. The `app=True` flag (or type inference) is the only thing you need.

## Mixing with Custom HTML Apps

Prefab tools and [custom HTML tools](/apps/low-level) coexist in the same server. Prefab tools share a single renderer resource; custom tools point to their own. Both use the same MCP Apps protocol:

```python theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.apps import AppConfig

@mcp.tool(app=True)
def team_directory() -> PrefabApp:
    ...

@mcp.tool(app=AppConfig(resource_uri="ui://my-app/map.html"))
def map_view() -> str:
    ...
```

## Next Steps

* **[Patterns](/apps/patterns)** — Charts, tables, forms, and other common tool UIs
* **[Custom HTML Apps](/apps/low-level)** — When you need your own HTML, CSS, and JavaScript
* **[Prefab UI Docs](https://prefab.prefect.io)** — Components, state, expressions, and actions
