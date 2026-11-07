"""Check camera thread boundaries and render-environment preflight binding."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/sorting_supervisor/scripts'))
import m4c_common


def manifest(environment):
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ,environment,clear=True))
        stack.enter_context(patch.object(m4c_common,'base_manifest',return_value={}))
        stack.enter_context(patch.object(m4c_common,'digest',return_value='bound-file'))
        stack.enter_context(patch.object(m4c_common,'get_package_prefix',return_value='/tmp/prefix'))
        stack.enter_context(patch.object(m4c_common,'executable',return_value=Path('/tmp/executable')))
        stack.enter_context(patch.object(m4c_common,'share',return_value=Path('/tmp/share')))
        return m4c_common.manifest()


class RenderingTests(unittest.TestCase):
    def test_camera_worker_never_calls_glfw(self):
        source=(ROOT/'src/sorting_simulation/src/traced_camera_plugin.cpp').read_text()
        for api in ('glfwInit(', 'glfwCreateWindow(', 'glfwMakeContextCurrent(', 'glfwDestroyWindow('):
            self.assertNotIn(api,source)

    def test_changed_backend_invalidates_preflight(self):
        automatic=manifest({})
        nvidia=manifest({'SORTING_RENDER_BACKEND':'nvidia'})
        self.assertNotEqual(automatic,nvidia)
        self.assertEqual(nvidia['m4c/render_backend_request'],'nvidia')

    def test_software_override_invalidates_preflight(self):
        hardware=manifest({'SORTING_RENDER_BACKEND':'nvidia'})
        software=manifest({'SORTING_RENDER_BACKEND':'nvidia','LIBGL_ALWAYS_SOFTWARE':'1'})
        self.assertNotEqual(hardware,software)
        self.assertTrue(software['m4c/software_rendering'])


if __name__=='__main__':unittest.main()
