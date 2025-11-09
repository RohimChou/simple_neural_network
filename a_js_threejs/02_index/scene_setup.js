import * as THREE from "https://esm.sh/three@0.181.0";
import CameraControls from "https://esm.sh/camera-controls@3.1.1";
CameraControls.install({ THREE });
import { setupLights } from "./lights.js";
import { getXYZPlanes } from "./xyz_plane.js";
import { getAxes } from "./axes.js";

export function setupScene(scene) {
    // set up lighting
    const lights = setupLights();
    scene.add(lights);

    // set up XYZ planes
    const xyzPlanes = getXYZPlanes();
    scene.add(xyzPlanes);

    // set up axes
    const axes = getAxes();
    scene.add(axes);

    // set up camera, renderer, and controls
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(2, 2, 4.5); // Set initial camera position

    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const orbitControls = new CameraControls(camera, document.body);
    const clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate); // Schedule the next frame
        orbitControls.update(clock.getDelta()); // Update the controls based on time elapsed
        renderer.render(scene, camera); // Render the scene from the perspective of the camera
    }
    animate(); // Start the animation loop

    return { camera, renderer, orbitControls };
}
