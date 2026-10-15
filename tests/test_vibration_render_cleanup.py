"""Execute the renderer's browser-side resource lifecycle without WebGL."""

import shutil
import subprocess

import pytest

from tooltoad.scene3d import (
    GridScene,
    MoleculeModel,
    Py3DmolGridRenderer,
    SceneCell,
    VibrationAnimation,
)


def test_export_installs_cleanup_before_starting_each_animation():
    scene = GridScene(
        cells=[SceneCell(
            models=[MoleculeModel(atoms=["H"], coords=[[0.0, 0.0, 0.0]])],
            animations=[VibrationAnimation(mode=[[0.1, 0.0, 0.0]], reps=0)],
        )],
        columns=1,
    )
    html = Py3DmolGridRenderer(scene).render().write_html()
    assert html.index("_tooltoadVibrationCleanup") < html.index(".animate(")
    assert "UNIQUEID" not in html
    assert '"reps": 0' in html


def test_replaced_frame_resources_and_picking_shapes_are_bounded():
    """Release replaced child resources once and keep unchanged frames alive."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the JavaScript lifecycle test")
    script = r'''
const assert = require('node:assert/strict');
let created = 0, disposed = 0, deletedBuffers = 0;
const atom = {clickable: true, intersectionShape: {sphere: []}};
const resource = () => ({
    geometryGroups: [{__webglRadiusBuffer: {}, __webglOffsetBuffer: {}}],
    dispose() { disposed++; }
});
const model = {atoms: [atom], molObj: null, renderedMolObj: null};
let pending = 0;
const timer = {cancel() { pending = 0; }};
const viewer = {
    animationTimers: new Set([timer]),
    resumeAnimate() { pending++; return this; },
    models: [model],
    renderer: {_gl: {deleteBuffer() { deletedBuffers++; }}},
    render(rebuild = true) {
        if (model.molObj === null) {
            atom.intersectionShape.sphere.push({});
            const geometry = resource(), material = resource();
            // Shared children must be released only once.
            model.renderedMolObj = {children: [
                {geometry, material}, {geometry, material}
            ]};
            model.molObj = model.renderedMolObj;
            created += 2;
        }
        assert.equal(atom.intersectionShape.sphere.length, 1);
        return this;
    }
};
INSTALL(viewer);
const wrapped = viewer.render;
INSTALL(viewer);
assert.equal(wrapped, viewer.render);
for (let i = 0; i < 100; i++) {
    assert.equal(viewer.resumeAnimate(), viewer);
    assert.equal(pending, 1);
}
for (let i = 0; i < 100; i++) {
    model.molObj = null;
    assert.equal(viewer.render(), viewer);
    const count = disposed;
    viewer.render();
    assert.equal(disposed, count);
    assert.equal(created - disposed, 2);
}
assert.equal(disposed, 198);
assert.equal(deletedBuffers, 198);
'''.replace("INSTALL", f"({Py3DmolGridRenderer._VIBRATION_RENDER_CLEANUP})")
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
