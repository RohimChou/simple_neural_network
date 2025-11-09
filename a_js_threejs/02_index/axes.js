import * as THREE from "https://esm.sh/three@0.181.0";
import SpriteText from "https://esm.sh/three-spritetext@1.10.0";

export function getAxes() {
    const labelsInfo = {
        'X': { position: new THREE.Vector3(3, 0, 0), color: 'white' },
        'Y': { position: new THREE.Vector3(0, 3, 0), color: 'white' },
        'Z': { position: new THREE.Vector3(0, 0, 3), color: 'white' }
    };

    const group = new THREE.Group();
    for (const key in labelsInfo) {
        const info = labelsInfo[key];
        group.add(createLabel(key, info.position, info.color));
    }

    // axesHelper - creates lines for the X, Y, and Z axes
    const axesHelper = new THREE.AxesHelper(25); // Size of the axes lines
    group.add(axesHelper);

    return group; 
}

function createLabel(text, position, color) {
    const label = new SpriteText(text);
    label.color = color;
    label.textHeight = 0.3;     // adjust to your scene scale
    label.position.copy(position);
    label.fontFace = 'Times New Roman, Times, serif';
    return label;
}