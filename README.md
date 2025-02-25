## Gaphor AutoLayout
This is a plugin for [Gaphor](https://github.com/gaphor/gaphor) that implements a more powerful auto layout engine using [ELKJS](https://github.com/kieler/elkjs/). 

## Installation
Install to Gaphor using guidelines here: https://docs.gaphor.org/en/latest/plugins.html

### Dependencies
- [NodeJS](https://nodejs.org/en) - This is expected to be installed in the default locations for various operating systems (via website or homebrew)

Run the following command to grab and install to gaphor via github. This has been tested with MacOS. It should also function for Linux.
```cmd
pip install --target $HOME/.local/gaphor/plugins-2 git+https://github.com/tompkins-ct/gaphor-autolayout.git && npm install --prefix $HOME/.local/gaphor/plugins-2/gaphor_autolayout git+https://github.com/tompkins-ct/gaphor-autolayout.git
```
