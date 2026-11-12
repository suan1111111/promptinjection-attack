from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.office_utils import call_office


class BusyError(Exception):
    hresult = -2147418111


class OfficeRetryTests(unittest.TestCase):
    def test_busy_office_is_pumped_then_retried(self):
        action = Mock(side_effect=[BusyError(), 'converted'])
        pump = Mock()
        with patch.dict(sys.modules, {'pythoncom': SimpleNamespace(PumpWaitingMessages=pump)}), patch('services.office_utils.time.sleep'):
            self.assertEqual(call_office(action), 'converted')
        self.assertEqual(action.call_count, 2)
        pump.assert_called_once()

    def test_non_busy_error_is_not_hidden_or_retried(self):
        action = Mock(side_effect=ValueError('bad document'))
        with self.assertRaisesRegex(ValueError, 'bad document'):
            call_office(action)
        action.assert_called_once()

    def test_busy_timeout_returns_the_actual_error(self):
        action = Mock(side_effect=BusyError())
        with self.assertRaises(BusyError):
            call_office(action, timeout=0)
        action.assert_called_once()
