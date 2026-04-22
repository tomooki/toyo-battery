"""Enable ``python -m toyo_battery.gui`` as a shortcut for the Tk entry point."""

from toyo_battery.gui.tk_app import launch_gui

if __name__ == "__main__":
    launch_gui()
