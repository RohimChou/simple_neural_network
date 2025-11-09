import * as THREE from "https://esm.sh/three@0.181.0";

/**
 * draw a point (sphere) at (x, y, z) with given color and size
 * 
 * @param {Number} x 
 * @param {Number} y 
 * @param {Number} z
 * @param {Number} color, e.g. 0xff0000 for red
 * @param {Number} size, radius of the sphere
 * @returns THREE.Mesh object representing the point
 */
export function plotPoint(x, y, z = 0, color = 0xff0000, size = 0.1, opacity = 0.9) {
    // widthSegments = number of vertical slices, like longitude lines
    // heightSegments = number of horizontal slices, like latitude lines
    // Segments define how smooth the sphere looks
    const geometry = new THREE.SphereGeometry(size, 20, 20);
    const material = new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: opacity });
    const point = new THREE.Mesh(geometry, material);
    point.position.set(x, y, z);
    return point;
}


/**
 * Draw a vector (arrow) from 'from' to 'to' with given color
 * 
 * @param {THREE.Vector3} from
 * @param {THREE.Vector3} to
 * @param {Number} color
 */
export function drawVector(from, to, color = 0xff0000) {
    const direction = new THREE.Vector3().subVectors(to, from).normalize();
    const length = from.distanceTo(to);
    const arrow = new THREE.ArrowHelper(direction, from, length, color);
    return arrow;
}

/**
 * Draw a thick vector (arrow with cylinder shaft and cone head) from 'from' to 'to' with given color
 *
 * @param {THREE.Vector3} from
 * @param {THREE.Vector3} to
 * @param {Number} color
 * @param {Number} shaftRadius
 * @param {Number} headRadius
 * @param {Number} headLengthRatio
 */
export function drawThickVector(from, to, color = 0xff0000, shaftRadius = 0.05, headRadius = 0.075, headLengthRatio = 0.15) {
    const dir = new THREE.Vector3().subVectors(to, from);
    const length = dir.length();
    dir.normalize();

    // Create an empty group to hold shaft + head
    const arrow = new THREE.Group();

    // Shaft (cylinder)
    const shaftLength = length * (1 - headLengthRatio);
    const shaftGeo = new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 16);
    const shaftMat = new THREE.MeshStandardMaterial({ color });
    const shaft = new THREE.Mesh(shaftGeo, shaftMat);

    // Move shaft so its bottom is at origin
    shaft.position.y = shaftLength / 2;
    arrow.add(shaft);

    // Head (cone)
    const headLength = length * headLengthRatio;
    const headGeo = new THREE.ConeGeometry(headRadius, headLength, 16);
    const head = new THREE.Mesh(headGeo, shaftMat);
    head.position.y = shaftLength + headLength / 2;
    arrow.add(head);

    // Orient arrow to match direction
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    // Move to start point
    arrow.position.copy(from);

    return arrow;
}