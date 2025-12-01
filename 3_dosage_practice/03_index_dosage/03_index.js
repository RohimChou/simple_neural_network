import * as THREE from "https://esm.sh/three@0.181.0";
import CameraControls from "https://esm.sh/camera-controls@3.1.1";
CameraControls.install({ THREE });
import SpriteText from "https://esm.sh/three-spritetext@1.10.0";

import { setupScene } from "./scene_setup.js";
import { plotPoint, drawVector, drawThickVector } from "./plotting_utils.js";

export function start() {
    const scene = new THREE.Scene();
    const { camera, renderer, orbitControls } = setupScene(scene);

    // test_datas = [
    //     TestData(0, 0),
    //     TestData(1, 0.2),
    //     TestData(3, 0.9),
    //     TestData(5, 1),
    //     TestData(7, 0.7),
    //     TestData(10, 0.1),
    // ]
    scene.add(plotPoint(0, 0, 0, 0xff0000, 0.08));
    scene.add(plotPoint(1, 0.2, 0, 0xff0000, 0.08));
    scene.add(plotPoint(3, 0.9, 0, 0xff0000, 0.08));
    scene.add(plotPoint(5, 1, 0, 0xff0000, 0.08));
    scene.add(plotPoint(7, 0.7, 0, 0xff0000, 0.08));
    scene.add(plotPoint(10, 0.1, 0, 0xff0000, 0.08));

    // // Visualize decision boundary
    // for (let x = -1.3; x <= 2; x += 0.08) {
    //     for (let y = -1.3; y <= 2; y += 0.08) {
    //         const z = trainedModel(x, y);
    //         scene.add(plotPoint(x, y, z, 0x888888, 0.02, 0.7));
    //     }
    // }

    // function trainedModel(x, y) {
    //     const layer1_weights = [[6.03761467, 4.32420299],
    //                             [6.02081736, 3.94247363]];
    //     const layer1_biases = [-2.54031315, -6.23497739];
    //     const layer2_weights = [-7.2634691, 7.69750747];
    //     const layer2_biases = [3.28919445];

    //     const layer1_output = [];
    //     layer1_output.push(layer1_weights[0][0] * x + layer1_weights[0][1] * y + layer1_biases[0]);
    //     layer1_output.push(layer1_weights[1][0] * x + layer1_weights[1][1] * y + layer1_biases[1]);
    //     layer1_output[0] = sigmoid(layer1_output[0]);
    //     layer1_output[1] = sigmoid(layer1_output[1]);

    //     let layer2_output = layer2_weights[0] * layer1_output[0] 
    //                       + layer2_weights[1] * layer1_output[1] + layer2_biases[0];
    //     layer2_output = sigmoid(layer2_output);
    //     return layer2_output;
    // }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }


    // // Draw thick arrows and animate with raycasting
    // const arrow0 = drawThickVector(
    //     new THREE.Vector3(0, 0, 0),
    //     new THREE.Vector3(0, 0, 2),
    //     0x00ff00
    // );
    // scene.add(arrow0);

    // const arrow = drawThickVector(
    //     new THREE.Vector3(0, 0, 0),
    //     new THREE.Vector3(1, 1, 0),
    //     0x0000ff
    // );
    // scene.add(arrow);


    // // Raycasting setup
    // const pickables = []; // store objects that can be picked
    // pickables.push(arrow0);
    // pickables.push(arrow);
    // const raycaster = new THREE.Raycaster();
    // const mouse = new THREE.Vector2();

    // window.addEventListener('pointermove', (event) => {
    //     mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    //     mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    // });


    // function animate() {
    //     requestAnimationFrame(animate);

    //     // Cast a ray from camera through mouse
    //     raycaster.setFromCamera(mouse, camera);
    //     // Intersect with all thick arrows
    //     const hits = raycaster.intersectObjects(pickables, true); // <— true = search in children
    //     // Reset all arrows to default color first
    //     pickables.forEach(arrow => {
    //         arrow.children.forEach(mesh => mesh.material.color.set(0xff0000)); // default color
    //     });
    //     // If something is hovered, highlight it
    //     if (hits.length > 0) {
    //         const hoveredArrow = hits[0].object.parent; // arrow group
    //         hoveredArrow.children.forEach(mesh => mesh.material.color.set(0x00ff00)); // highlight color
    //     }

    //     arrow.rotation.z -= 0.001; // rotation around the arrow's local Z axis clockwise
    //     renderer.render(scene, camera);
    // }
    // animate();
}
