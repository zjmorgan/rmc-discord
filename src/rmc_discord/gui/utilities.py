"""Background workers (as in NeuXtalViz), with task output sent to the console."""

import logging
import sys
import threading
import traceback

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    progress = Signal(str, int)
    result = Signal(object)
    output = Signal(str)


class _ThreadStream:
    """
    ``sys.stdout``/``sys.stderr`` stand-in that sends text written by a
    worker thread to that worker's console and everything else to the
    original stream (redirecting globally would mix concurrent workers).
    """

    def __init__(self, stream):
        self.stream = stream
        self.sinks = {}

    def write(self, s):
        sink = self.sinks.get(threading.get_ident())
        if sink is not None:
            if s:
                sink(s)
            return len(s)
        return self.stream.write(s) if self.stream is not None else len(s)

    def flush(self):
        if self.stream is not None and threading.get_ident() not in self.sinks:
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _ThreadLogHandler(logging.Handler):
    def __init__(self, sinks):
        super().__init__(logging.INFO)
        self.sinks = sinks

    def emit(self, record):
        sink = self.sinks.get(threading.get_ident())
        if sink is not None:
            sink(self.format(record))


_lock = threading.Lock()
_streams = None


def _install():
    global _streams
    with _lock:
        if _streams is None:
            out, err = _ThreadStream(sys.stdout), _ThreadStream(sys.stderr)
            sys.stdout, sys.stderr = out, err
            logging.getLogger().addHandler(_ThreadLogHandler(out.sinks))
            _streams = (out, err)
    return _streams


class Worker(QRunnable):
    """
    Run a task in the thread pool.

    The task receives a ``progress(message, percent)`` callback as the
    keyword argument ``progress``; what it prints (or logs) appears in the
    view's console.
    """

    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.signals = WorkerSignals()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.kwargs["progress"] = self.emit_progress

    @Slot()
    def run(self):
        streams = _install()
        ident = threading.get_ident()
        for s in streams:
            s.sinks[ident] = self.signals.output.emit
        try:
            result = self.task(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            for s in streams:
                s.sinks.pop(ident, None)
            self.signals.finished.emit()

    def emit_progress(self, message, progress):
        self.signals.progress.emit(message, progress)

    def connect_result(self, process):
        self.signals.result.connect(process)

    def connect_finished(self, process):
        self.signals.finished.connect(process)

    def connect_progress(self, process):
        self.signals.progress.connect(process)

    def connect_error(self, process):
        self.signals.error.connect(process)

    def connect_output(self, process):
        self.signals.output.connect(process)


class ThreadPool(QThreadPool):
    """
    Thread pool that keeps its workers alive until they finish.

    Without a Python reference the worker (and the QObject carrying its
    signals) can be garbage collected while it runs, and its queued
    ``result``/``finished`` signals are then never delivered: the task
    seems to hang and its buttons stay disabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workers = set()

    def start_worker_pool(self, worker):
        worker.setAutoDelete(False)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.start(worker)
