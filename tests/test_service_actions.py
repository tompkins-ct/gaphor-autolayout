import pytest

from gaphor.core.eventmanager import EventManager
from gaphor.core.modeling.elementdispatcher import ElementDispatcher
from gaphor.core.modeling.modelinglanguage import (
    CoreModelingLanguage,
    MockModelingLanguage,
)
from gaphor.diagram.general.modelinglanguage import GeneralModelingLanguage
from gaphor.SysML.modelinglanguage import SysMLModelingLanguage
from gaphor.UML.modelinglanguage import UMLModelingLanguage
from gaphor.core import Transaction
from gaphor.core.modeling import ElementFactory
from gaphor.UML.uml import Diagram

from gaphor_autolayout.autolayoutelk import (
    AutoLayoutELKService,
    layout_properties_normal,
)


class FakeDiagrams:
    def __init__(self, diagram):
        self._diagram = diagram

    def get_current_diagram(self):
        return self._diagram


class FakeDialog:
    def __init__(self, to_return: dict | None):
        self._to_return = to_return
        self.received_initial = None

    def open(self, initial_props: dict | None = None):
        self.received_initial = initial_props
        return self._to_return


@pytest.fixture
def event_manager():
    return EventManager()


@pytest.fixture
def modeling_language():
    return MockModelingLanguage(
        CoreModelingLanguage(),
        GeneralModelingLanguage(),
        UMLModelingLanguage(),
        SysMLModelingLanguage(),
    )


@pytest.fixture
def element_factory(event_manager, modeling_language):
    element_factory = ElementFactory(
        event_manager, ElementDispatcher(event_manager, modeling_language)
    )
    yield element_factory
    element_factory.shutdown()


@pytest.fixture
def diagram(element_factory, event_manager):
    with Transaction(event_manager):
        diagram = element_factory.create(Diagram)
    yield diagram
    with Transaction(event_manager):
        diagram.unlink()


def test_open_custom_properties_initializes_with_normal_and_stores(diagram, event_manager):
    fake_dialog = FakeDialog({"elk.algorithm": "layered", "custom": "yes"})
    service = AutoLayoutELKService(event_manager, FakeDiagrams(diagram), tools_menu=None, dump_gv=False)
    # inject dialog (will be added by implementation)
    service.layout_properties_dialog = fake_dialog

    # ACT: open the dialog via action
    service.open_custom_layout_properties()

    # ASSERT: dialog received initial = layout_properties_normal()
    assert fake_dialog.received_initial == layout_properties_normal()
    # and service stored the returned custom props (whatever the dialog returned)
    assert service._custom_layout_properties == fake_dialog._to_return


def test_apply_custom_properties_calls_layout_with_custom(diagram, event_manager, monkeypatch):
    fake_props = {"elk.algorithm": "layered", "elk.direction": "DOWN", "custom": 1}
    fake_dialog = FakeDialog(fake_props)
    service = AutoLayoutELKService(event_manager, FakeDiagrams(diagram), tools_menu=None, dump_gv=False)
    service.layout_properties_dialog = fake_dialog

    # Simulate user opening and accepting dialog
    service.open_custom_layout_properties()

    called = {}

    def fake_layout(diag, props):
        called["diagram"] = diag
        called["props"] = props

    # Monkeypatch the instance method
    monkeypatch.setattr(service, "layout", fake_layout)

    # ACT: apply via action
    service.apply_custom_layout_properties()

    # ASSERT: layout called with current diagram and stored custom props
    assert called["diagram"] is diagram
    assert called["props"] == fake_props
