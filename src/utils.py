import serial
from serial.tools.list_ports import comports


def get_serial(port, *args, **kwargs):
    """
    A wrapper method for opening serial port with automatic port listing and
    automatic/interactive selection.
    """
    if port is not None:
        return serial.Serial(port, *args, **kwargs)
    else:
        ports = tuple(comports())  # list detected serial ports
    if len(ports) == 0:
        raise Exception("No serial ports found!")
    elif len(ports) == 1:
        port = ports[0]
    else:
        port = None
        while port is None:
            sel = int(input(
                "\nAvailable ports:\n{}\n\nSelect port to use: ".format(
                    '\n'.join([
                        str(i) + ": " + x.device
                        for i, x in enumerate(ports)]))))
            if sel >= 0 and sel < len(ports):
                port = ports[sel]
    print("\nUsing port: {}".format(port.device))
    return serial.Serial(port.device, *args, **kwargs)
