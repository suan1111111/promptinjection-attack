import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.ocr_service import convert_office_to_pdf


@unittest.skipUnless(os.name == 'nt', 'Windows Office adapter')
class OfficeConversionTests(unittest.TestCase):
    def run_conversion(self, existing=0, fail=False):
        document = SimpleNamespace(Close=Mock())

        def save(path, file_format):
            if fail:
                raise RuntimeError('export failed')
            Path(path).write_bytes(b'%PDF-test')

        document.SaveAs = Mock(side_effect=save)
        app = SimpleNamespace(Presentations=SimpleNamespace(Count=existing, Open=Mock(return_value=document)), Quit=Mock())
        com = SimpleNamespace(CoInitializeEx=Mock(), COINIT_APARTMENTTHREADED=2, CoUninitialize=Mock())
        win32 = ModuleType('win32com')
        win32.client = SimpleNamespace(DispatchEx=Mock(return_value=app))
        temp_root = Path(__file__).resolve().parents[2] / 'tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory, patch.dict(sys.modules, {
            'pythoncom': com, 'win32com': win32, 'win32com.client': win32.client,
        }):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'export failed'):
                    convert_office_to_pdf(str(Path(directory) / 'fixture.pptx'), directory)
            else:
                self.assertTrue(Path(convert_office_to_pdf(str(Path(directory) / 'fixture.pptx'), directory)).exists())
        document.Close.assert_called_once_with()
        com.CoInitializeEx.assert_called_once_with(2)
        com.CoUninitialize.assert_called_once()
        return app

    def test_existing_presentations_are_not_closed_with_the_application(self):
        self.run_conversion(existing=1).Quit.assert_not_called()

    def test_failed_export_still_closes_owned_document_and_application(self):
        self.run_conversion(fail=True).Quit.assert_called_once()
