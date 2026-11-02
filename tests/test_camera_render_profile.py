from pathlib import Path
import importlib.util
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src/sorting_perception/scripts"))
from m3a_probe import CameraProbe

ROOT=Path(__file__).resolve().parents[1]

class ProfileTests(unittest.TestCase):
    def test_generator_preserves_the_fast_profile_and_camera_geometry(self):
        spec=importlib.util.spec_from_file_location("camera_scene_generator",ROOT/"scripts/prepare_m3a_scene.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);description=root/"src/sorting_description";perception=root/"src/sorting_perception"
            base=description/"mujoco/static_grasp";base.mkdir(parents=True);(perception/"config").mkdir(parents=True)
            simulation=root/"src/sorting_simulation/config";simulation.mkdir(parents=True)
            originals={}
            for name in ("scene_tracking.xml","panda_static.xml","props.xml"):
                shutil.copy2(ROOT/"src/sorting_description/mujoco/static_grasp"/name,base/name)
                originals[name]=(base/name).read_bytes()
            shutil.copy2(ROOT/"src/sorting_perception/config/cameras.yaml",perception/"config/cameras.yaml")
            shutil.copy2(ROOT/"src/sorting_simulation/config/conveyor_plugins.yaml",simulation/"conveyor_plugins.yaml")
            with patch.multiple(module,ROOT=root,DESCRIPTION=description,PERCEPTION=perception):module.main()
            scene=ET.parse(base/"scene_camera.xml").getroot();quality=scene.find("visual/quality")
            self.assertEqual(quality.get("offsamples"),"4");self.assertEqual(quality.get("shadowsize"),"512")
            self.assertTrue(all(light.get("castshadow")=="false" for light in scene.findall(".//light")))
            self.assertEqual(scene.find("worldbody/camera[@name='overhead']").get("resolution"),"320 240")
            panda=ET.parse(base/"panda_camera.xml").getroot()
            self.assertEqual(panda.find(".//camera[@name='wrist']").get("resolution"),"320 240")
            self.assertTrue(all(light.get("castshadow")=="false" for light in panda.findall(".//light")))
            for name,content in originals.items():self.assertEqual((base/name).read_bytes(),content)
    def test_image_age_and_position_error_limits_are_not_relaxed(self):
        c=yaml.safe_load((ROOT/"src/sorting_perception/config/m3a.yaml").read_text())
        self.assertEqual(c["maximum_image_age_s"],.15);self.assertEqual(c["maximum_planar_error_m"],.002)

class WaitTests(unittest.TestCase):
    def test_actual_failed_wrist_wait_reports_counts_and_latency(self):
        f=json.loads((ROOT/"tests/fixtures/stationary-camera-wait.json").read_text())
        c=yaml.safe_load((ROOT/"src/sorting_perception/config/m3a.yaml").read_text())
        p=SimpleNamespace(c=c,phase=f["phase"],phase_begin_ns=f["phase_begin_ns"],frames=f["frames"],
                          camera_status=f["camera_status"],camera_wait_diagnostic=None)
        p.phase_samples=lambda view:[s for s in f["samples"] if s["view_id"]==view and s["capture_ns"]>p.phase_begin_ns]
        def expired(*args):raise TimeoutError("Ownership probe timed out")
        p.until=expired
        with self.assertRaisesRegex(TimeoutError,"view=wrist, paired=1/10"):
            CameraProbe.collect(p,"wrist",10)
        d=p.camera_wait_diagnostic
        self.assertEqual(d["paired_samples"],1);self.assertEqual(d["required_samples"],10)
        self.assertGreater(d["median_received_image_age_s"],.15)
        self.assertGreater(d["localizer_rejection_reasons"]["capture_age_outside_bound"],200)
    def test_completed_wait_returns_the_measured_samples(self):
        rows=[{"capture_ns":1},{"capture_ns":2}]
        p=SimpleNamespace(phase_samples=lambda view:rows,until=lambda predicate,timeout:self.assertTrue(predicate()))
        self.assertEqual(CameraProbe.collect(p,"wrist",2),rows)

if __name__=="__main__":unittest.main()
