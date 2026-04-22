"""Enable ``python -m echemplot.gui`` as a shortcut for the Tk entry point."""

from echemplot.gui.tk_app import launch_gui

if __name__ == "__main__":
    launch_gui()
