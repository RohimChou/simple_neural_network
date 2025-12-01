import * as THREE from "https://esm.sh/three@0.181.0";

export function setupLights() {
    const group = new THREE.Group();
    
    const ambientLight = new THREE.AmbientLight(0x555555); // Soft white light
    group.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight.position.set(7, 7, 7).normalize();
    group.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 2);
    directionalLight2.position.set(-7, -7, -7).normalize();
    group.add(directionalLight2);

    return group;
}
