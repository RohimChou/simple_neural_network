import * as THREE from "https://esm.sh/three@0.181.0";

/**
 * Creates grid helpers for the XY, XZ, and YZ planes.
 * @returns {Object} An object containing the grid helpers for each plane.
 */
export function getXYZPlanes() {
    // creates a grid on the XZ plane (ground)
    const xyPlane = new THREE.GridHelper(
        50,        // Size of the grid. if size = 50, the grid spans 50 units across.
        150,       // Number of divisions. if divisions = 50, there will be 50 divisions across the grid.
        0x5f5f5f,  // Center line color
        0x444444   // Grid line color
    );
    xyPlane.material.opacity = 0.9;
    xyPlane.material.transparent = true;
    xyPlane.rotation.x = Math.PI / 2; // Rotate to lie on XY plane
    xyPlane.scale.set(1, 1, 3); // Scale to make grid lines denser along X axis

    const xzPlane = new THREE.GridHelper(
        50,
        50,
        0x445f44,
        0x224422
    );
    xzPlane.material.opacity = 0.9;
    xzPlane.material.transparent = true;

    const yzPlane = new THREE.GridHelper(
        50,
        50,
        0x5f445f,
        0x442244
    );
    yzPlane.material.opacity = 0.9;
    yzPlane.material.transparent = true;
    yzPlane.rotation.z = Math.PI / 2; // Rotate to lie on YZ plane

    const group = new THREE.Group();
    group.add(xyPlane);
    group.add(xzPlane);
    group.add(yzPlane);
    return group;
}