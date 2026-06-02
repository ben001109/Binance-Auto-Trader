import contextlib
import io
import unittest
from unittest.mock import patch

from bat.config import Config


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type

    def __str__(self) -> str:
        return self.type


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool, built: bool = True):
        self._available = available
        self._built = built

    def is_available(self) -> bool:
        return self._available

    def is_built(self) -> bool:
        return self._built


class _FakeXpu:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(
        self,
        *,
        cuda: bool = False,
        hip: str | None = None,
        mps: bool = False,
        xpu: bool = False,
    ):
        self.cuda = _FakeCuda(cuda)
        self.backends = type("Backends", (), {"mps": _FakeMps(mps)})()
        self.version = type("Version", (), {"hip": hip})()
        self.xpu = _FakeXpu(xpu)

    def device(self, device_type: str) -> _FakeDevice:
        return _FakeDevice(device_type)


class DeviceSelectionTest(unittest.TestCase):
    def _select_device(self, fake_torch: _FakeTorch):
        stream = io.StringIO()
        with patch("bat.config.torch", fake_torch), contextlib.redirect_stdout(stream):
            device = Config()._get_optimal_device()
        return device, stream.getvalue()

    def test_rocm_cuda_api_is_reported_as_amd_rocm(self):
        device, output = self._select_device(_FakeTorch(cuda=True, hip="6.1.0"))

        self.assertEqual(device.type, "cuda")
        self.assertIn("AMD ROCm", output)

    def test_intel_xpu_is_used_when_available(self):
        device, output = self._select_device(_FakeTorch(xpu=True))

        self.assertEqual(device.type, "xpu")
        self.assertIn("Intel XPU", output)


if __name__ == "__main__":
    unittest.main()
