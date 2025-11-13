[![CI](https://github.com/tompkins-ct/gaphor-autolayout/actions/workflows/ci.yml/badge.svg)](https://github.com/tompkins-ct/gaphor-autolayout/actions/workflows/ci.yml)

## Gaphor AutoLayout
This is a plugin for [Gaphor](https://github.com/gaphor/gaphor) that implements a more powerful auto layout engine using [ELKJS](https://github.com/kieler/elkjs/). 

## Install
This package is a gaphor plugin. 
Refer Gaphor plugin: https://docs.gaphor.org/en/latest/plugins.html

### Dependencies
- [NodeJS](https://nodejs.org/en) - Required for ELKJS
- [Elkjs](https://github.com/kieler/elkjs) - Layout engine

### Plugin Installation

Run the following command to grab and install to gaphor via github. 
This has been tested with MacOS and Ubuntu.
```cmd
pip install --target $HOME/.local/gaphor/plugins-2 git+https://github.com/tompkins-ct/gaphor-autolayout.git && npm install --prefix $HOME/.local/gaphor/plugins-2/gaphor_autolayout git+https://github.com/tompkins-ct/gaphor-autolayout.git
```

### Package Installation
```cmd
npm install elkjs
pip install git+https://github.com/tompkins-ct/gaphor-autolayout.git
```

## Usage
- Plugin may be used in the gaphor UI via the tools menu after installation

When using as a standalone package for use with gaphor programmatically, import and use as follows:
```python
from gaphor_autolayout.autolayoutelk import AutoLayoutELK, layout_properties_normal, layout_properties_topdown
from gaphor.core.modeling import ElementFactory
from gaphor import UML

element_factory = ElementFactory()

diagram = element_factory.create(UML.Diagram)

autolayout = AutoLayoutELK()

# left-to-right layout, default
autolayout.layout(diagram, layout_props=layout_properties_normal())

# top-to-bottom layout
autolayout.layout(diagram, layout_props=layout_properties_topdown())
```

The `layout_props` parameter is an optional `dict` and may be used to specify the layout algorithm settings.
Refer to the ELK [documentation](https://eclipse.dev/elk/reference.html) for more details on the available options.


## Contributing

PRs accepted.
