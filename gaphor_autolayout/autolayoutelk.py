import json
import os
import subprocess
import logging
import typing
import asyncio

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import singledispatch

from gaphas.connector import ConnectionSink, Connector
from gaphas.geometry import Point
from gaphas.item import NW
from gaphas.matrix import Matrix
from gaphas.segment import Segment
from gaphor.UML import Comment

from gaphor.abc import ActionProvider, Service
from gaphor.action import action
from gaphor.core.modeling import Base, Diagram, Presentation
from gaphor.diagram.group import can_group, group
from gaphor.diagram.presentation import (
    AttachedPresentation,
    ElementPresentation,
    LinePresentation,
)
from gaphor.i18n import gettext
from gaphor.transaction import Transaction

# UML specific imports (could probably avoid)
from gaphor.UML.actions.activitynodes import ForkNodeItem
from gaphor.UML.classes import AssociationItem, DependencyItem, GeneralizationItem

log = logging.getLogger(__name__)

@dataclass
class BaseItem:
    """Base item type for JSON output"""

    id: str
    properties: Mapping[str, object]


@dataclass
class SizedElement(BaseItem):
    """Base type for non-line like items in the graph"""

    x: float | None
    y: float | None
    width: float
    height: float
    labels: list


@dataclass
class Node(SizedElement):
    """Node JSON definition"""

    ports: list
    children: list
    edges: list


class Port(SizedElement):
    """Port JSON definition"""

    pass


@dataclass
class Label(SizedElement):
    """Label JSON definition"""

    text: str


@dataclass
class Edge(BaseItem):
    """Edge JSON definition"""

    sources: list[Node]
    targets: list[Node]
    sections: list
    labels: list[Label | dict[str, str | int | float | None]]


@dataclass
class Sections:
    """Edge section definitions for JSON"""

    start_point: dict[int, int]
    end_point: dict[int, int]
    bend_point: list
    incomingShape: str
    outgoingShape: str
    incomingSections: list
    outgoingSections: list


async def open_elk_properties_dialog(properties: dict) -> dict | None:
    """Open a dialog to configure custom ELK layout properties.

    This function opens a small Libadwaita (Adw/Gtk4) dialog that allows the user
    to inspect and edit key/value pairs in a dictionary. Users can:
      - Edit existing keys and values
      - Add new rows
      - Remove rows
      - Reset to the initially provided values

    Behavior notes and fallbacks:
      - The function is "best effort": if GI/GTK is not available, or no display
        can be opened, the function logs a warning and returns ``None`` without
        raising. This keeps headless test environments working.
      - Values are parsed using ``json.loads`` when possible, so numbers,
        booleans, ``null`` (None), arrays and objects are supported. If parsing
        fails, the raw string is used.
      - Duplicate keys are allowed while editing, but when applying the result
        the last (bottom-most) occurrence wins.

    The function is ``async`` and will offload the modal GTK loop to a worker
    thread using ``asyncio.to_thread`` so it does not block the caller's event
    loop.
    """

    log = logging.getLogger(__name__)

    initial = dict(properties or {})

    def _parse_value(text: str):
        text = text if text is not None else ""
        # Strip surrounding whitespace; allow unquoted barewords to be treated
        # as strings unless they parse as JSON literals
        try:
            return json.loads(text)
        except Exception:
            return text

    def _format_value(value):
        # Provide a readable string in the entry for non-strings
        if isinstance(value, (int, float)):
            return str(value)
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "null"
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except Exception:
                return str(value)
        return str(value)

    def _run_dialog_sync() -> dict | None:
        try:
            import os as _os
            # Require GI at runtime only.
            import gi  # type: ignore

            try:
                gi.require_version("Adw", "1")
                gi.require_version("Gtk", "4.0")
            except Exception:
                # Older gi may use "Gtk", "4.0"; attempt anyway
                pass

            from gi.repository import Adw, Gtk, GLib  # type: ignore
        except Exception as e:  # pragma: no cover - GUI not available in tests
            log.warning("Libadwaita/Gtk not available for properties dialog: %s", e)
            return None

        # If there is no display (headless), bail out gracefully
        if not _os.environ.get("DISPLAY") and not _os.environ.get("WAYLAND_DISPLAY"):
            log.info("No DISPLAY/WAYLAND_DISPLAY set; skipping properties dialog.")
            return None

        # Ensure Adwaita styling is initialized
        Adw.init()

        result_container: dict[str, object] | None = None

        # Build UI
        window = Adw.Window()
        window.set_title("ELK Layout Properties")
        window.set_default_size(520, 420)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_top=12, margin_bottom=12, margin_start=12, margin_end=12)
        window.set_content(vbox)

        # Header with actions
        header = Adw.HeaderBar()
        vbox.append(header)

        add_button = Gtk.Button.new_from_icon_name("list-add-symbolic")
        add_button.set_tooltip_text("Add property")
        header.pack_start(add_button)

        reset_button = Gtk.Button.new_with_mnemonic("_Reset")
        reset_button.set_tooltip_text("Reset to initial values")
        header.pack_start(reset_button)

        # Main list
        scrolled = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        vbox.append(scrolled)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        scrolled.set_child(listbox)

        def _add_row(k: str = "", v: object = ""):
            row = Gtk.ListBoxRow()
            hb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            row.set_child(hb)

            key_entry = Gtk.Entry()
            key_entry.set_hexpand(True)
            key_entry.set_placeholder_text("key")
            key_entry.set_text(k or "")

            value_entry = Gtk.Entry()
            value_entry.set_hexpand(True)
            value_entry.set_placeholder_text("value (JSON or text)")
            value_entry.set_text(_format_value(v))

            remove_btn = Gtk.Button.new_from_icon_name("user-trash-symbolic")
            remove_btn.set_tooltip_text("Remove this property")

            def _remove(_btn):
                listbox.remove(row)

            remove_btn.connect("clicked", _remove)

            hb.append(Gtk.Label(label="Key", xalign=0))
            hb.append(key_entry)
            hb.append(Gtk.Label(label="Value", xalign=0))
            hb.append(value_entry)
            hb.append(remove_btn)

            # Store for collection later
            row._key_entry = key_entry  # type: ignore[attr-defined]
            row._value_entry = value_entry  # type: ignore[attr-defined]

            listbox.append(row)

        # Populate with initial properties
        for k, v in initial.items():
            _add_row(str(k), v)

        # Add a blank row for convenience
        _add_row()

        def _on_add(_btn):
            _add_row()

        add_button.connect("clicked", _on_add)

        def _on_reset(_btn):
            # Clear all rows and repopulate
            for child in list(listbox):  # type: ignore[arg-type]
                listbox.remove(child)
            for k, v in initial.items():
                _add_row(str(k), v)
            _add_row()

        reset_button.connect("clicked", _on_reset)

        # Action buttons at the bottom
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        action_box.set_halign(Gtk.Align.END)
        vbox.append(action_box)

        cancel_btn = Gtk.Button.new_with_mnemonic("_Cancel")
        apply_btn = Gtk.Button.new_with_mnemonic("_Apply")
        action_box.append(cancel_btn)
        action_box.append(apply_btn)

        loop = GLib.MainLoop()

        def _collect_and_close(accept: bool):
            nonlocal result_container
            if accept:
                new_props: dict[str, object] = {}
                # Iterate all rows and build dict; last occurrence wins
                for row in listbox:  # type: ignore[assignment]
                    try:
                        key = row._key_entry.get_text().strip()  # type: ignore[attr-defined]
                        val_text = row._value_entry.get_text()  # type: ignore[attr-defined]
                    except Exception:
                        continue
                    if key:
                        new_props[key] = _parse_value(val_text)
                result_container = new_props
            else:
                result_container = None
            loop.quit()
            window.close()

        cancel_btn.connect("clicked", lambda _b: _collect_and_close(False))
        apply_btn.connect("clicked", lambda _b: _collect_and_close(True))
        window.connect("close-request", lambda *_: (_collect_and_close(False), True)[1])

        window.present()
        try:
            loop.run()
        finally:
            try:
                loop.quit()
            except Exception:
                pass
        return result_container

    # Run the blocking GTK loop in a thread so we can await it cleanly
    try:
        return await asyncio.to_thread(_run_dialog_sync)
    except Exception as e:  # pragma: no cover - safety net
        log.exception("Failed to open ELK properties dialog: %s", e)
        return None

class AutoLayoutELKService(Service, ActionProvider):
    """Service provider for Autolayout using ELK"""

    def __init__(self, event_manager, diagrams, tools_menu=None, dump_gv=False):
        self.event_manager = event_manager
        self.diagrams = diagrams
        if tools_menu:
            tools_menu.add_actions(self)
        self.dump_gv = dump_gv

        # Storage for user-configured custom properties
        self._custom_layout_properties: dict = layout_properties_normal()

    def shutdown(self):
        pass

    @action(
        name="auto-layout-ELK_LR",
        label=gettext("Auto Layout ELK Left-to-Right"),
        # shortcut="<Primary><Shift>L"
    )
    def layout_current_diagram_lr(self):
        if current_diagram := self.diagrams.get_current_diagram():
            layout_props = layout_properties_normal()
            self.layout(current_diagram, layout_props)

    @action(
        name="auto-layout-ELK-TD",
        label=gettext("Auto Layout ELK Top-to-Bottom"),
        # shortcut="<Primary><Shift>L"
    )
    def layout_current_diagram_td(self):
        if current_diagram := self.diagrams.get_current_diagram():
            layout_props = layout_properties_topdown()
            self.layout(current_diagram, layout_props)

    @action(
        name="auto-layout-ELK-custom-open",
        label=gettext("Auto Layout ELK: Configure Custom Properties…"),
    )
    def open_custom_layout_properties(self):
        """Open a dialog to configure custom ELK layout properties.

        The dialog (if provided) is initialized with `layout_properties_normal()`
        and should return a dict with properties or None if canceled.
        """

        initial = self._custom_layout_properties or layout_properties_normal()

        result: dict | None = None

        # If a dialog provider was injected (mainly for tests), use it
        dlg = getattr(self, "layout_properties_dialog", None)
        if dlg and hasattr(dlg, "open"):
            try:
                result = dlg.open(initial_props=initial)
            except TypeError:
                # Backward-compat without keyword
                result = dlg.open(initial)
        else:
            # Fall back to the built-in async dialog and run it to completion
            try:
                try:
                    # If no loop is running, run the coroutine directly
                    loop = asyncio.get_running_loop()
                    running = loop.is_running()
                except RuntimeError:
                    running = False
                if not running:
                    result = asyncio.run(open_elk_properties_dialog(initial))
                else:
                    # If we're already inside an event loop, this action should not block.
                    # In that case, skip opening the dialog to avoid deadlocks.
                    log.info("Event loop running; skipping synchronous properties dialog.")
                    result = None
            except Exception as e:
                log.exception("Failed to obtain custom layout properties: %s", e)
                result = None

        if isinstance(result, dict):
            self._custom_layout_properties = result
        else:
            log.info("Custom layout properties dialog cancelled or returned no data.")

    @action(
        name="auto-layout-ELK-custom-apply",
        label=gettext("Auto Layout ELK: Custom"),
    )
    def apply_custom_layout_properties(self):
        """Apply previously configured custom properties to the current diagram."""
        if current_diagram := self.diagrams.get_current_diagram():
            props = self._custom_layout_properties or layout_properties_normal()
            self.layout(current_diagram, props)

    def layout(self, diagram: Diagram, layout_props):
        auto_layout = AutoLayoutELK(self.event_manager)

        with Transaction(self.event_manager):
            auto_layout.layout(diagram, layout_props)


def layout_properties_normal() -> dict:
    """Setup properties for the top level of the diagram"""
    properties = {
        "elk.algorithm": "layered",
        "elk.layered.feedbackEdges": "true",  # feedback edges loop around the layout
        "org.eclipse.elk.hierarchyHandling": "INCLUDE_CHILDREN",  # allows edges to move between layers
        "elk.layoutHierarchy": "true",  # enables routing between layers
        "elk.edgeRouting": "ORTHOGONAL",  # explict default
        "elk.nodeLabels.placement": "H_CENTER V_TOP INSIDE",  # nominal gaphor placement for node labels
        "elk.nodeSize.constraints": "MINIMUM_SIZE_ACCOUNTS_FOR_PADDING",  # allows for resizing of nodes
        "org.eclipse.elk.layered.spacing.edgeNodeBetweenLayers": "25.0",  # edges in layer (room for arrow heads)
        "org.eclipse.elk.spacing.nodeSelfLoop": "20.0",  # space for arrows on self-loops,
        "org.eclipse.elk.font.size": "12",  # default font size for labels (not sure if this does anything)
    }
    return properties


def layout_properties_topdown() -> dict:
    """Setup properties for the top level of the diagram"""
    properties = {
        "elk.algorithm": "layered",
        "elk.layered.feedbackEdges": "true",  # feedback edges loop around the layout
        "org.eclipse.elk.hierarchyHandling": "INCLUDE_CHILDREN",  # allows edges to move between layers
        "elk.layoutHierarchy": "true",  # enables routing between layers
        "elk.edgeRouting": "ORTHOGONAL",  # explict default
        "elk.nodeLabels.placement": "H_CENTER V_TOP INSIDE",  # nominal gaphor placement for node labels
        "elk.nodeSize.constraints": "MINIMUM_SIZE_ACCOUNTS_FOR_PADDING",  # allows for resizing of nodes
        "org.eclipse.elk.layered.spacing.edgeNodeBetweenLayers": "20.0",  # layer to layer placement
        "org.eclipse.elk.spacing.nodeSelfLoop": "20.0",  # space for arrows on self-loops,
        "org.eclipse.elk.font.size": "12",  # default font size for labels (not sure if this does anything)
        "elk.direction": "DOWN",
        "bk.fixedAlignment": "BALANCED", # shifts vertical alignment of layer to even edge path lengths.
    }
    return properties


def baseline_graph(properties) -> Node:
    """Defines top level diagram node"""
    return Node(
        id="graph",
        properties=properties,
        x=0.0,
        y=0.0,
        width=10.0,
        height=10.0,
        labels=[],
        ports=[],
        children=[],
        edges=[],
    )


class AutoLayoutELK:
    """Autolayout diagram"""

    def __init__(self, event_manager=None) -> None:
        self.event_manager = event_manager
        self.graph: Node | None = None

    def layout(self, diagram: Diagram, layout_props: typing.Optional[dict] = None) -> None:
        """Generate the layout from ELKjs"""
        diagram.update(diagram.ownedPresentation)
        # in the future, adjust layout properties based on the diagram type
        if layout_props is None:
            layout_props = layout_properties_normal()

        self.graph = baseline_graph(layout_props)
        self.generate_graph(diagram)

        # render graph using ELKjs engine
        json_export = self.convert_graph()
        log.info(f"Exported layout graph: {json_export}")
        current_directory = os.path.dirname(os.path.abspath(__file__))
        elkjs_runner = os.path.join(current_directory, "elkrunner.js")
        rendered_graph_as_str = _run_nodejs_script(elkjs_runner, [json_export])
        log.info(f"Elk rendered graph {rendered_graph_as_str}")
        rendered_graph_as_dict = json.loads(rendered_graph_as_str)

        # get resulting node locations for use late
        node_positions: dict[str, tuple[float, float]] = {}
        _get_positions(rendered_graph_as_dict, node_positions)

        self.apply_layout(diagram, rendered_graph_as_dict, node_positions)
        diagram.update(diagram.ownedPresentation)

    def generate_graph(self, diagram: Diagram):
        """Generate a graph using the supplied diagram"""
        for presentation in diagram.ownedPresentation:
            if (
                presentation.parent
            ):  # skipping items with parents (e.g., addressed recursively instead)
                continue
            _add_to_graph(self.graph, as_graph(presentation))

    def convert_graph(self):
        """ELKjs uses a JSON file to define the graph structure and parameters."""
        return json.dumps(self.graph, default=lambda o: getattr(o, "__dict__", str(o)))

    def apply_layout(
        self,
        diagram: Diagram,
        updated_layout: dict,
        node_positions: dict,
        parent_presentation=None,
        height=None,
    ):
        """Apply the ELKjs layout to the Gaphor diagram."""
        if height is None:
            height = updated_layout["height"]

        # ELK uses the top left corner of the parent node as the positional reference.

        # define a translation matrix for dealing with translation relative to parent node
        # i2c is the matrix relation to the top level of the diagram
        # matrix on its own is relation to the parent
        matrix_c2i = (
            parent_presentation.matrix_i2c.inverse()
            if parent_presentation
            else Matrix()
        )

        for node in updated_layout["children"]:
            if presentation := _presentation_for_object(diagram, node):
                if isinstance(presentation, ElementPresentation):
                    presentation.handles()[NW].pos = (0.0, 0.0)
                    presentation.width = node["width"]
                    presentation.height = node["height"]
                    presentation.matrix.set(x0=node["x"], y0=node["y"])

                    if (
                        can_group(parent_presentation, presentation)
                        and parent_presentation is not None
                    ):
                        group(parent_presentation, presentation)

                    if len(node["children"]) > 0 or len(node["ports"]) > 0:
                        self.apply_layout(
                            diagram,
                            node,
                            node_positions,
                            presentation,
                            height,
                        )
                else:  # deal with node-like objects that are not boxes (forkNodes...)
                    new_pos = matrix_c2i.transform_point(node["x"], node["y"])
                    presentation.matrix.set(
                        x0=new_pos[0],
                        y0=new_pos[1],
                    )

        # ports
        for port in updated_layout["ports"]:
            if presentation := _presentation_for_object(diagram, port):
                if isinstance(presentation, AttachedPresentation):
                    # Elkjs location is the top left corner relative to the owner's top left corner.
                    # Gaphor uses the center for attached items.
                    presentation.handles()[NW].pos = (0.0, 0.0)
                    presentation.matrix.set(
                        x0=port["x"] + presentation.width / 2,
                        y0=port["y"] + presentation.height / 2,
                    )
                    _reconnect(
                        presentation, presentation.handles()[0], diagram.connections
                    )

        for edge in updated_layout["edges"]:
            # ELK edges all have at least 1 segment with a start point and end point
            # if there are bends, they are added as a series of bendpoints via section definitions.
            if presentation := _presentation_for_object(diagram, edge):
                presentation.orthogonal = False

                # Generalizations are drawn backwards relatively to other items
                reverse = isinstance(presentation, GeneralizationItem)

                # ELK defines locations relative to the containing node so they need to be adjusted to absolute
                relative_location = _get_relative_location_from_container(
                    edge, node_positions, diagram
                )

                # edges are defined relative to the container
                points = _parse_edge_pos(edge["sections"], relative_location, reverse)
                segment = Segment(presentation, diagram)


                # setting the number of handles equal to the number of points
                while len(points) > len(presentation.handles()):
                    try:
                        segment.split_segment(0)
                    except ValueError:
                        log.error(f"Spitting {edge} failed.")
                        raise ValueError(
                            f"Cannot split with {len(points)} segments. Edge {edge} failed."
                        )
                while len(points) < len(presentation.handles()):
                    try:
                        segment.merge_segment(0)
                    except ValueError:
                        log.error(f"Merging {edge} failed.")
                        raise ValueError(f"Cannot merge with 1 segment. Edge {edge} failed.")


                assert len(points) == len(presentation.handles())

                # apply points to the line.
                matrix = presentation.matrix_i2c.inverse()
                for handle, point in zip(presentation.handles(), points, strict=False):
                    handle.pos = matrix.transform_point(*point)

                for handle in (presentation.head, presentation.tail):
                    _reconnect(presentation, handle, diagram.connections)


def _add_to_graph(parent, edge_or_node) -> None:
    """recursively add elements to the graph from the diagram"""
    if isinstance(edge_or_node, Edge):
        parent.edges.append(edge_or_node)
    elif isinstance(edge_or_node, Node):
        parent.children.append(edge_or_node)
    elif isinstance(edge_or_node, Port):
        parent.ports.append(edge_or_node)
    elif isinstance(edge_or_node, Iterable):
        for obj in edge_or_node:
            _add_to_graph(parent, obj)
    elif edge_or_node:
        raise ValueError(
            f"Can't recognize input item {edge_or_node} for addition to the graph."
        )


def _run_nodejs_script(script_path, arg):
    """run Node.js script from python"""
    # Note: the path in compiled bytecode is different from straight run so we need to find the NodeJS executable
    if os.path.exists("/usr/local/bin/node"):
        node_exc = r"/usr/local/bin/node"
    elif os.path.exists("/opt/homebrew/bin/node"):
        node_exc = r"/opt/homebrew/bin/node"
    elif os.path.exists("C:/Program Files/nodejs"):
        node_exc = r"C:/Program Files/nodejs.exe"
    else:
        raise Exception("Can't find nodejs executable")

    cmd = [node_exc, script_path] + arg
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        # cannot find node app
        raise Exception(
            "Error: NodeJS was not found. Check PATH or use absolute path. Current PATH:",
            os.environ["PATH"],
        )

    if result.returncode == 0:
        return result.stdout
    else:
        raise Exception(f"Error running or finding Node.js script: {result.stderr} with the following input: {arg[0]}")


def _as_cluster(presentation: Presentation):
    """Check if all children are elements (not ports)"""
    # return presentation.children and not all(
    #     isinstance(c, AttachedPresentation) for c in presentation.children
    # )
    if len(presentation.children) > 0:
        return True
    else:
        return False


def _get_relative_location_from_container(edge, node_positions, diagram: Diagram):
    """Determine relative change based on the container's top left corner for edges"""
    relative_location = [0.0, 0.0]
    if edge["container"] != "graph":
        owner_id = edge["container"]
        x_pos, y_pos = node_positions[owner_id]
        parent_container: Presentation = next(
            (p for p in diagram.ownedPresentation if p.id == owner_id), None
        )
        if parent_container is not None:
            while parent := parent_container.parent:  # type: ignore[union-attr]
                parent_id = parent.id
                p_coord_x, p_coord_y = node_positions[parent_id]
                x_pos += p_coord_x
                y_pos += p_coord_y
                parent_container = next(
                    (p for p in diagram.ownedPresentation if p.id == parent_id), None
                )
        relative_location = [x_pos, y_pos]

    return relative_location


def _get_relative_location_from_parent(parent_presentation: Presentation | None):
    """Get the location of the parent presentation for relative positioning."""
    if parent_presentation is None:
        return [0.0, 0.0]
    _, _, _, _, x_pos, y_pos = parent_presentation.matrix
    relative_location = [x_pos, y_pos]
    return relative_location


def _presentation_for_object(diagram: Diagram, obj) -> Presentation | LinePresentation | None:
    """Get the diagram presentation for a given object"""
    if not obj.get("id"):
        return None

    id = _strip_quotes(obj.get("id"))
    return next((p for p in diagram.ownedPresentation if p.id == id), None)


def _reconnect(presentation, handle, connections) -> None:
    """Reconnect objects to their end point elements"""
    if not (connected := connections.get_connection(handle)):
        return

    connector = Connector(presentation, handle, connections)
    sink = ConnectionSink(connected.connected, distance=float("inf"))
    connector.connect(sink)


def _strip_quotes(s):
    """Replace quotes and line continuations in the read JSON string"""
    return s.replace('"', "").replace("\\\n", "").replace("\\\r\n", "")


def _parse_edge_pos(
    sections: dict, relative_location: list[float], reverse: bool
) -> list[Point]:
    """Handle points are defined relative to the containing node and adjusted accordingly"""
    points = []
    for section in sections:
        point_dict = section["startPoint"]
        points.append(
            (
                point_dict["x"] + relative_location[0],
                point_dict["y"] + relative_location[1],
            )
        )
        try:
            for bend in section["bendPoints"]:
                points.append(
                    (bend["x"] + relative_location[0], bend["y"] + relative_location[1])
                )
        except KeyError:
            pass
        point_dict = section["endPoint"]
        points.append(
            (
                point_dict["x"] + relative_location[0],
                point_dict["y"] + relative_location[1],
            )
        )

    if reverse:
        points.reverse()
    return points


@singledispatch
def as_graph(element: Base) -> Iterator[BaseItem]:
    """Process items to yield Nodes, Ports, or Edges"""
    return iter(())


@as_graph.register
def _(presentation: ElementPresentation):
    # check if the element contains other elements (nested)
    if _as_cluster(presentation):
        # create node for primary element
        min_width = presentation.min_width.value  # type: ignore[union-attr]
        min_height = presentation.min_height.value  # type: ignore[union-attr]
        node_layout_options = {
            "elk.nodeLabels.placement": "[H_CENTER V_TOP INSIDE]",
            "elk.padding": "[left=10, top=30, right=10, bottom=30]",
            "org.eclipse.elk.nodeSize.minimum": (
                "(" + str(min_width) + ", " + str(min_height) + ")"
            ),
            "org.eclipse.elk.nodeSize.constraints": "MINIMUM_SIZE",
        }

        # comments can be placed out special (allow for more flexibility as opposed to the left to right)
        if isinstance(presentation, Comment):
            node_layout_options["org.eclipse.elk.commentBox"] = "true"

        label_layout_options: dict[str, str | int | float | None] = {}

        # Can get label width here. However, node labels wrap and minimum size is established by Gaphor

        graph = Node(
            id=presentation.id,
            properties=node_layout_options,
            x=None,
            y=None,
            width=presentation.width,
            height=presentation.height,
            ports=[],
            children=[],
            edges=[],
            labels=[
                Label(
                    id=presentation.id + "_label",
                    properties=label_layout_options,
                    x=None,
                    y=None,
                    width=50,
                    height=30,
                    labels=[],
                    text=_get_label(presentation),
                ),
            ],  # add label because this is a containing node that needs space for the label
        )

        # add children to new node
        for child in presentation.children:
            _add_to_graph(graph, as_graph(child))

        # return the top-leve node w/ its defined children
        yield graph
    else:  # single element
        # I don't understand what this is protecting against.
        # If there is a single element, then it should be a node

        # check if the node is a comment
        node_layout_options = {}
        if isinstance(presentation, Comment):
            node_layout_options["org.eclipse.elk.commentBox"] = "true"

        yield Node(
            id=presentation.id,
            properties=node_layout_options,
            x=None,
            y=None,
            width=presentation.width,
            height=presentation.height,
            ports=[],
            children=[],
            edges=[],
            labels=[],
        )


@as_graph.register
def _(presentation: LinePresentation):
    connections = presentation.diagram.connections
    head_connection = connections.get_connection(presentation.head)
    tail_connection = connections.get_connection(presentation.tail)
    layout_properties = {}

    if isinstance(presentation, GeneralizationItem):
        layout_properties["org.eclipse.elk.edge.type"] = "GENERALIZATION"
    elif isinstance(presentation, AssociationItem):
        layout_properties["org.eclipse.elk.edge.type"] = "ASSOCIATION"
    elif isinstance(presentation, DependencyItem):
        layout_properties["org.eclipse.elk.edge.type"] = "DEPENDENCY"
    else:
        layout_properties["org.eclipse.elk.edge.type"] = "ASSOCIATION"

    if (
        head_connection
        and next(as_graph(head_connection.connected), None)
        and tail_connection
        and next(as_graph(tail_connection.connected), None)
    ):
        # extra_args = {}
        # if as_cluster(head_connection.connected):
        #     extra_args["lhead"] = f"cluster_{head_connection.connected.id}"
        # if as_cluster(tail_connection.connected):
        #     extra_args["ltail"] = f"cluster_{tail_connection.connected.id}"

        # swap direction due to hierarchy for generalizations (do we do the same for composition?)
        if isinstance(presentation, GeneralizationItem):
            head_id = tail_connection.connected.id
            tail_id = head_connection.connected.id
        else:
            head_id = head_connection.connected.id
            tail_id = tail_connection.connected.id

        # get edge label and calculated width needed
        edge_label = _get_label(presentation)

        yield Edge(
            id=presentation.id,
            properties=layout_properties,
            sources=[head_id],
            targets=[tail_id],
            labels=[
                Label(
                    id=presentation.id + "_l0",
                    properties={},
                    x=None,
                    y=None,
                    text=edge_label,
                    width=_get_label_width(edge_label),
                    height=12.0,
                    labels=[],
                )
            ],
            sections=[],
        )


@as_graph.register
def _(presentation: AttachedPresentation):
    """process port-type items

    Note: endpoints from ELK are located at the center of ports and causes connection position issue
    Note2: included labels so space is provided should labels exist on the ports.
    """
    port_properties = {
        "org.eclipse.elk.port.borderOffset": -presentation.width / 2.0,
        "org.eclipse.elk.portLabels.placement": "OUTSIDE",
    }
    # port_properties["org.eclipse.elk.port.anchor"] = "(8.0, 0.0)"
    # using "proxy" to label port because it is the default for most ports.
    yield Port(
        id=presentation.id,
        properties=port_properties,
        x=0.0,
        y=0.0,
        width=presentation.width,
        height=presentation.height,
        labels=[
            Label(
                id=presentation.id + "_l0",
                properties={},
                x=None,
                y=None,
                text="proxy",
                width=_get_label_width("proxy"),
                height=15.0,
                labels=[],
            )
        ],
    )


@as_graph.register
def _(presentation: ForkNodeItem):
    """process port-type items"""
    h1, h2 = presentation.handles()
    yield Node(
        id=presentation.id,
        properties={},
        x=0.0,
        y=0.0,
        width=h2.pos.x - h1.pos.x,
        height=h2.pos.y - h1.pos.y,
        ports=[],
        children=[],
        edges=[],
        labels=[],
    )


def _get_label_width(label: str) -> float:
    return len(label) * 8.0  # assuming 0.5 em width and 12 pt font


def _get_label(presentation: Presentation) -> str:
    """Helper function"""
    try:
        return (
            presentation.subject.name if presentation.subject.name is not None else " "
        )
    except AttributeError:
        return " "


def _get_positions(layout: dict, position_dict: dict):
    """Get node positions for use in auto-layout"""
    for node in layout["children"]:
        position_dict[node["id"]] = (node["x"], node["y"])
        if len(node["children"]) > 0:
            _get_positions(node, position_dict)
