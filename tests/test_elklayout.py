import json
import logging
import pytest

from gaphor.application import Application

from gaphor import UML
from gaphor_autolayout.autolayoutelk import (
    AutoLayoutELK,
    _parse_edge_pos,
    _strip_quotes, layout_properties_normal,
)

from gaphor.UML.diagramitems import (
    ActionItem,
    AssociationItem,
    ClassItem,
    ForkNodeItem,
    GeneralizationItem,
    InputPinItem,
    ObjectFlowItem,
    PackageItem,
)
from gaphor.UML.general import CommentItem, CommentLineItem
from gaphor.diagram.presentation import connect as _connect
from gaphor.core import Transaction
from gaphor.core.eventmanager import EventManager
from gaphor.core.modeling import ElementFactory
from gaphor.core.modeling.elementdispatcher import ElementDispatcher
from gaphor.core.modeling.modelinglanguage import (
    CoreModelingLanguage,
    MockModelingLanguage,
)
from gaphor.diagram.general.modelinglanguage import GeneralModelingLanguage
from gaphor.SysML.modelinglanguage import SysMLModelingLanguage
from gaphor.UML.modelinglanguage import UMLModelingLanguage
from gaphor.UML.uml import Diagram


log = logging.getLogger(__name__)


def connect(line, handle, item, port=None):
    """Connect line's handle to an item.

    If a port is not provided, then the first port is used.
    """
    _connect(line, handle, item)

    cinfo = line.diagram.connections.get_connection(handle)
    assert cinfo.connected is item
    assert cinfo.port


@pytest.fixture
def application():
    app = Application()
    yield app
    app.shutdown()


@pytest.fixture
def session(application):
    return application.new_session()


@pytest.fixture
def event_manager():
    return EventManager()


@pytest.fixture
def element_factory(event_manager, modeling_language):
    element_factory = ElementFactory(
        event_manager, ElementDispatcher(event_manager, modeling_language)
    )
    yield element_factory
    element_factory.shutdown()


@pytest.fixture
def modeling_language():
    return MockModelingLanguage(
        CoreModelingLanguage(),
        GeneralModelingLanguage(),
        UMLModelingLanguage(),
        SysMLModelingLanguage(),
    )


@pytest.fixture
def diagram(element_factory, event_manager):
    with Transaction(event_manager):
        diagram = element_factory.create(Diagram)
    yield diagram
    with Transaction(event_manager):
        diagram.unlink()


@pytest.fixture
def create(diagram, element_factory):
    def _create(item_class, element_class=None):
        return diagram.create(
            item_class,
            subject=(element_factory.create(element_class) if element_class else None),
        )

    return _create


def test_layout_diagram(diagram, create):
    superclass = create(ClassItem, UML.Class)
    subclass = create(ClassItem, UML.Class)
    gen = create(GeneralizationItem, UML.Generalization)
    connect(gen, gen.tail, superclass)
    connect(gen, gen.head, subclass)

    auto_layout = AutoLayoutELK()
    auto_layout.layout(diagram)

    assert gen.head.pos != (0, 0)
    assert gen.tail.pos != (0, 0)


def test_layout_with_association(diagram, create, event_manager):
    c1 = create(ClassItem, UML.Class)
    c2 = create(ClassItem, UML.Class)
    a = create(AssociationItem)
    connect(a, a.head, c1)
    connect(a, a.tail, c2)

    layout_props = layout_properties_normal()
    auto_layout = AutoLayoutELK()
    auto_layout.layout(diagram, layout_props)


# def test_layout_with_comment(diagram, create, event_manager):
#     """Failure due to comment line connecting to an edge.
#
#     How to treat? May need to link to a node (such as the edge label?,
#     e.g., is source/target is an edge then connect to the associated label)"""
#     c1 = create(ClassItem, UML.Class)
#     c2 = create(ClassItem, UML.Class)
#     a = create(AssociationItem)
#     connect(a, a.head, c1)
#     connect(a, a.tail, c2)
#
#     comment = create(CommentItem, UML.Comment)
#     comment_line = create(CommentLineItem)
#     connect(comment_line, comment_line.head, comment)
#     connect(comment_line, comment_line.tail, a)
#
#     auto_layout = AutoLayoutELK(event_manager)
#     auto_layout.layout(diagram)


def test_layout_with_nested(diagram, create, event_manager):
    p = create(PackageItem, UML.Package)
    c1 = create(ClassItem, UML.Class)
    p.children = c1
    c2 = create(ClassItem, UML.Class)
    a = create(AssociationItem)
    connect(a, a.head, c1)
    connect(a, a.tail, c2)

    layout_props = layout_properties_normal()
    auto_layout = AutoLayoutELK()
    auto_layout.layout(diagram, layout_props)

    assert c1.matrix[4] < p.width
    assert c1.matrix[5] < p.height


def test_layout_with_attached_item(diagram, create, event_manager):
    action = create(ActionItem, UML.Action)
    pin = create(InputPinItem, UML.InputPin)
    connect(pin, pin.handles()[0], action)

    action2 = create(ActionItem, UML.Action)
    object_flow = create(ObjectFlowItem, UML.ObjectFlow)
    connect(object_flow, object_flow.head, pin)
    connect(object_flow, object_flow.tail, action2)

    layout_props = layout_properties_normal()
    auto_layout = AutoLayoutELK()
    auto_layout.layout(diagram, layout_props)

    assert pin.parent is action


def test_layout_fork_node_item(diagram, create, event_manager):
    create(ForkNodeItem, UML.ForkNode)

    auto_layout = AutoLayoutELK(event_manager)
    auto_layout.layout(diagram)


def test_parse_pos():
    json_test_string = '{"sections": [{"startPoint": {"x": 1.0, "y": 2.0}, "endPoint": {"x": 3.0, "y": 4.0}}]}'
    section_list = json.loads(json_test_string)
    relative_location = [0.0, 0.0]
    points = _parse_edge_pos(section_list["sections"], relative_location, False)

    assert points == [(1.0, 2.0), (3.0, 4.0)]


def test_strip_line_endings():
    assert _strip_quotes("\\\n807.5") == "807.5"
    assert _strip_quotes("\\\r\n807.5") == "807.5"
    assert _strip_quotes('\\\r\n"807.5"') == "807.5"
