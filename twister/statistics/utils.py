
"""Utils functions."""
import logging
import multiprocessing
import multiprocessing.pool

L = logging.getLogger(__name__)


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


def timeout_eval(func, args, timeout=None, pool=None):
    """Evaluate a function and kill it is it takes longer than timeout.

    If timeout is Nonei or == 0, a simple evaluation will take place.
    """
    if timeout is None or timeout == 0:
        return func(*args)

    return pool.apply_async(func, args).get(timeout=timeout)