import unittest

from bat.tui.app import CryptoApp


class _FakeTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


class TuiShutdownTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_cancels_model_watch_without_training_task(self):
        app = CryptoApp.__new__(CryptoApp)
        app.background_training_task = None
        app.model_watch_task = _FakeTask()
        app.download_worker = None
        app.simulation_future = None
        app.train_collect_future = None

        await app.on_shutdown()

        self.assertTrue(app.model_watch_task.cancelled)


if __name__ == "__main__":
    unittest.main()
