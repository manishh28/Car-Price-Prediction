import hashlib
import html
import json
import re
from string import Template


PAINT_COLOR = "#18786d"
BODY_STYLES = {
    "coupe",
    "hatchback",
    "mpv",
    "pickup",
    "sedan",
    "suv",
    "van",
    "wagon",
}


def _contains_term(name: str, terms: tuple[str, ...]) -> bool:
    return any(term in name for term in terms)


def vehicle_body_style(model: str, seats: int = 5) -> str:
    name = " ".join(re.sub(r"[^a-z0-9]+", " ", model.lower()).split())

    if _contains_term(
        name,
        (
            "bolero pik up",
            "camper",
            "isuzu d max",
            "isuzu v cross",
            "pickup",
            "pik up",
            "tata xenon",
        ),
    ):
        return "pickup"

    if _contains_term(
        name,
        (
            "ashok leyland",
            "eeco",
            "force traveller",
            "force winger",
            "maruti omni",
            "minivan",
            "van",
        ),
    ):
        return "van"

    if _contains_term(
        name,
        (
            "br v",
            "brv",
            "carens",
            "ertiga",
            "evalia",
            "enjoy",
            "innova",
            "lodgy",
            "marazzo",
            "mobilio",
            "renault triber",
            "tavera",
            "versa",
            "xylo",
        ),
    ):
        return "mpv"

    if _contains_term(
        name,
        (
            "aria",
            "audi q",
            "bolero",
            "captiva",
            "compass",
            "creta",
            "cross",
            "duster",
            "ecosport",
            "endeavour",
            "escudo",
            "evoque",
            "fortuner",
            "freelander",
            "harrier",
            "hexa",
            "hyundai santa fe",
            "jeep",
            "kicks",
            "koleos",
            "kuv",
            "land rover",
            "lexus es",
            "mahindra scorpio",
            "mahindra thar",
            "mercedes benz gl",
            "mercedes benz ml",
            "mg hector",
            "mitsubishi montero",
            "mitsubishi outlander",
            "mitsubishi pajero",
            "nissan terrano",
            "nuvosport",
            "quanto",
            "range rover",
            "renault captur",
            "renault duster",
            "safari",
            "santa fe",
            "seltos",
            "sonet",
            "sumo",
            "suv",
            "terrano",
            "thar",
            "tucson",
            "vitara brezza",
            "volvo xc",
            "xuv",
            "toyota urban cruiser",
            "bmw x1",
            "bmw x3",
            "bmw x4",
            "bmw x5",
            "bmw x6",
            "bmw x7",
        ),
    ):
        return "suv"

    if _contains_term(
        name,
        (
            "a star",
            "alto",
            "baleno",
            "beat",
            "brio",
            "celerio",
            "datsun go",
            "etios liva",
            "eon",
            "figo",
            "getz",
            "glanza",
            "grand i10",
            "i10",
            "i20",
            "ignis",
            "indica",
            "kwid",
            "micra",
            "nano",
            "polo",
            "pulse",
            "redi go",
            "ritz",
            "santro",
            "spark",
            "swift",
            "tiago",
            "wagon r",
        ),
    ):
        return "hatchback"

    if _contains_term(
        name,
        (
            "audi tt",
            "bmw z4",
            "coupe",
            "jaguar f type",
            "mercedes benz sl",
        ),
    ):
        return "coupe"

    if _contains_term(
        name,
        (
            "estate",
            "station wagon",
            "touring",
            "volvo v40",
        ),
    ):
        return "wagon"

    if seats >= 7:
        return "mpv"
    return "sedan"


def vehicle_shape_seed(model: str) -> int:
    digest = hashlib.blake2s(model.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def render_vehicle_preview(
    brand: str,
    model: str,
    year: int,
    fuel: str,
    transmission: str,
    matching_records: int,
    seats: int,
) -> str:
    body_style = vehicle_body_style(model, seats)
    config = {
        "bodyStyle": body_style,
        "model": model,
        "paint": PAINT_COLOR,
        "seed": vehicle_shape_seed(model),
    }
    config_json = json.dumps(config, ensure_ascii=True).replace("</", "<\\/")

    return VEHICLE_PREVIEW_TEMPLATE.substitute(
        brand=html.escape(brand),
        model=html.escape(model),
        year=int(year),
        fuel=html.escape(str(fuel)),
        transmission=html.escape(str(transmission)),
        matching_records=f"{int(matching_records):,}",
        body_style=body_style,
        paint=PAINT_COLOR,
        config_json=config_json,
    )


VEHICLE_PREVIEW_TEMPLATE = Template(
    r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
    :root {
        color-scheme: light;
        font-family: "Source Sans Pro", Arial, sans-serif;
    }

    * {
        box-sizing: border-box;
        letter-spacing: 0;
    }

    html, body {
        background: transparent;
        margin: 0;
        overflow: hidden;
        width: 100%;
    }

    .vehicle-stage {
        background: #102724;
        border-bottom: 4px solid #e7604f;
        color: #ffffff;
        display: grid;
        grid-template-columns: minmax(260px, 35%) minmax(0, 65%);
        height: 334px;
        overflow: hidden;
        width: 100%;
    }

    .vehicle-copy {
        align-content: center;
        display: grid;
        min-width: 0;
        overflow: hidden;
        padding: 28px 30px;
        position: relative;
        z-index: 2;
    }

    .vehicle-kicker {
        color: #8fc2b9;
        font-size: 12px;
        font-weight: 750;
        text-transform: uppercase;
    }

    .vehicle-name {
        color: #ffffff;
        font-size: 22px;
        font-weight: 750;
        line-height: 1.2;
        margin: 8px 0 18px;
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    .vehicle-facts {
        color: #c4d5d1;
        display: grid;
        font-size: 13px;
        gap: 9px;
    }

    .vehicle-facts strong {
        color: #ffffff;
        font-weight: 700;
    }

    .vehicle-view {
        background: #dfe7e5;
        min-width: 0;
        overflow: hidden;
        position: relative;
    }

    .vehicle-view canvas {
        display: block;
        height: 100%;
        outline: 0;
        width: 100%;
    }

    .vehicle-status {
        color: #52635f;
        font-size: 13px;
        left: 50%;
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
    }

    .vehicle-status.is-hidden {
        display: none;
    }

    @media (max-width: 680px) {
        .vehicle-stage {
            grid-template-columns: minmax(0, 1fr);
            grid-template-rows: 220px 340px;
            height: 564px;
        }

        .vehicle-copy {
            align-content: start;
            padding: 23px 20px;
        }

        .vehicle-name {
            font-size: 20px;
            margin-bottom: 14px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .vehicle-view canvas {
            scroll-behavior: auto;
        }
    }
</style>
</head>
<body>
<section class="vehicle-stage" aria-label="3D vehicle preview" data-body-style="$body_style" data-paint="$paint">
    <div class="vehicle-copy">
        <div class="vehicle-kicker">Selected $brand profile</div>
        <div class="vehicle-name">$model</div>
        <div class="vehicle-facts">
            <div><strong>$year</strong> typical listing year</div>
            <div><strong>$fuel</strong> &middot; <strong>$transmission</strong></div>
            <div><strong>$matching_records</strong> exact-match records</div>
        </div>
    </div>
    <div class="vehicle-view" id="vehicle-view">
        <div class="vehicle-status" id="vehicle-status">Preparing 3D preview...</div>
    </div>
</section>

<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js";

const config = $config_json;
const view = document.getElementById("vehicle-view");
const status = document.getElementById("vehicle-status");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function sendHeight() {
    const height = window.innerWidth <= 680 ? 564 : 334;
    window.parent.postMessage(
        {isStreamlitMessage: true, type: "streamlit:setFrameHeight", height},
        "*"
    );
}

sendHeight();
window.addEventListener("resize", sendHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe7e5);
scene.fog = new THREE.Fog(0xdfe7e5, 13, 24);

const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 60);
camera.position.set(6.2, 3.1, 7.5);
camera.lookAt(0.25, 0.85, 0);

const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: "high-performance"});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.setAttribute("aria-label", "Interactive 3D preview of " + config.model);
renderer.domElement.tabIndex = 0;
view.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x52635f, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 4.2);
keyLight.position.set(5, 8, 7);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xb9ddff, 2.0);
fillLight.position.set(-6, 3, -5);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffd8bf, 1.6);
rimLight.position.set(-2, 5, 7);
scene.add(rimLight);

const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(30, 30),
    new THREE.MeshStandardMaterial({color: 0xd8e1df, roughness: 0.93, metalness: 0})
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const styleSpecs = {
    sedan:    {length: 4.75, width: 1.82, roof: 1.68, ride: 0.42, wheel: 0.48, cabin: [-0.20, 0.27, 0.95]},
    hatchback:{length: 4.18, width: 1.78, roof: 1.73, ride: 0.43, wheel: 0.47, cabin: [-0.34, 0.27, 1.00]},
    suv:      {length: 4.82, width: 1.94, roof: 1.94, ride: 0.57, wheel: 0.57, cabin: [-0.37, 0.27, 1.06]},
    mpv:      {length: 4.72, width: 1.87, roof: 1.93, ride: 0.48, wheel: 0.52, cabin: [-0.41, 0.32, 1.06]},
    coupe:    {length: 4.58, width: 1.88, roof: 1.58, ride: 0.39, wheel: 0.50, cabin: [-0.16, 0.24, 0.90]},
    wagon:    {length: 4.78, width: 1.84, roof: 1.75, ride: 0.43, wheel: 0.49, cabin: [-0.36, 0.27, 0.98]},
    van:      {length: 4.75, width: 1.90, roof: 2.05, ride: 0.49, wheel: 0.51, cabin: [-0.43, 0.37, 1.08]},
    pickup:   {length: 5.08, width: 1.92, roof: 1.84, ride: 0.58, wheel: 0.58, cabin: [-0.08, 0.32, 1.01]}
};

const base = styleSpecs[config.bodyStyle] || styleSpecs.sedan;
const randomA = ((config.seed >>> 1) % 997) / 997;
const randomB = ((config.seed >>> 11) % 991) / 991;
const spec = {...base};
spec.length *= 0.97 + randomA * 0.06;
spec.width *= 0.98 + randomB * 0.04;

const paint = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(config.paint),
    clearcoat: 0.9,
    clearcoatRoughness: 0.16,
    metalness: 0.72,
    roughness: 0.24
});
const glass = new THREE.MeshStandardMaterial({
    color: 0x081518,
    metalness: 0.38,
    roughness: 0.14
});
const rubber = new THREE.MeshStandardMaterial({color: 0x111615, roughness: 0.82});
const alloy = new THREE.MeshStandardMaterial({color: 0xb7c2c0, metalness: 0.88, roughness: 0.2});
const dark = new THREE.MeshStandardMaterial({color: 0x17201f, metalness: 0.35, roughness: 0.38});
const headlight = new THREE.MeshPhysicalMaterial({color: 0xeaf7ff, emissive: 0xbddfff, emissiveIntensity: 0.65});
const taillight = new THREE.MeshPhysicalMaterial({color: 0xa51f2d, emissive: 0x5f0710, emissiveIntensity: 0.8});

function extrudedProfile(points, depth, material, bevel = 0.06) {
    const shape = new THREE.Shape();
    const last = points[points.length - 1];
    const first = points[0];
    shape.moveTo((last[0] + first[0]) / 2, (last[1] + first[1]) / 2);
    for (let i = 0; i < points.length; i += 1) {
        const point = points[i];
        const next = points[(i + 1) % points.length];
        shape.quadraticCurveTo(
            point[0],
            point[1],
            (point[0] + next[0]) / 2,
            (point[1] + next[1]) / 2
        );
    }
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, {
        depth,
        bevelEnabled: true,
        bevelSegments: 3,
        bevelSize: bevel,
        bevelThickness: bevel,
        curveSegments: 6
    });
    geometry.translate(0, 0, -depth / 2);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
}

const car = new THREE.Group();
car.rotation.y = -0.08;
car.position.y = 0.03;
scene.add(car);

const L = spec.length;
const W = spec.width;
const wheelY = spec.wheel;
const shoulder = spec.ride + 0.54;

const lowerBody = extrudedProfile([
    [-L * 0.50, wheelY * 0.62],
    [-L * 0.48, shoulder * 0.88],
    [-L * 0.37, shoulder],
    [ L * 0.36, shoulder],
    [ L * 0.49, shoulder * 0.82],
    [ L * 0.51, wheelY * 0.64]
], W, paint, 0.09);
car.add(lowerBody);

const cabinStart = spec.cabin[0] * L;
const cabinEnd = spec.cabin[1] * L;
const cabinRoof = spec.cabin[2] * spec.roof;
const outerCabin = extrudedProfile([
    [cabinStart, shoulder * 0.94],
    [cabinStart + L * 0.08, cabinRoof * 0.94],
    [cabinEnd - L * 0.08, cabinRoof],
    [cabinEnd, shoulder * 0.97]
], W * 0.89, paint, 0.08);
car.add(outerCabin);

const glassCabin = extrudedProfile([
    [cabinStart + L * 0.035, shoulder + 0.06],
    [cabinStart + L * 0.095, cabinRoof * 0.90],
    [cabinEnd - L * 0.095, cabinRoof * 0.95],
    [cabinEnd - L * 0.035, shoulder + 0.06]
], W * 0.905, glass, 0.035);
car.add(glassCabin);

const roofStrip = new THREE.Mesh(
    new THREE.BoxGeometry((cabinEnd - cabinStart) * 0.68, 0.075, W * 0.83),
    paint
);
roofStrip.position.set((cabinStart + cabinEnd) / 2, cabinRoof + 0.015, 0);
roofStrip.castShadow = true;
car.add(roofStrip);

if (config.bodyStyle === "pickup") {
    const bed = new THREE.Mesh(new THREE.BoxGeometry(L * 0.34, 0.34, W * 0.92), paint);
    bed.position.set(-L * 0.33, shoulder + 0.03, 0);
    bed.castShadow = true;
    car.add(bed);
}

const axleX = L * (config.bodyStyle === "van" ? 0.31 : 0.34);
for (const x of [-axleX, axleX]) {
    for (const z of [-W * 0.505, W * 0.505]) {
        const tire = new THREE.Mesh(
            new THREE.CylinderGeometry(spec.wheel, spec.wheel, 0.28, 36, 1),
            rubber
        );
        tire.rotation.x = Math.PI / 2;
        tire.position.set(x, wheelY, z);
        tire.castShadow = true;
        car.add(tire);

        const hub = new THREE.Mesh(
            new THREE.CylinderGeometry(spec.wheel * 0.57, spec.wheel * 0.57, 0.292, 12),
            alloy
        );
        hub.rotation.x = Math.PI / 2;
        hub.position.set(x, wheelY, z);
        car.add(hub);

        const cap = new THREE.Mesh(
            new THREE.CylinderGeometry(spec.wheel * 0.12, spec.wheel * 0.12, 0.306, 24),
            dark
        );
        cap.rotation.x = Math.PI / 2;
        cap.position.set(x, wheelY, z);
        car.add(cap);

        for (let spokeIndex = 0; spokeIndex < 5; spokeIndex += 1) {
            const spoke = new THREE.Mesh(
                new THREE.BoxGeometry(spec.wheel * 0.12, spec.wheel * 0.8, 0.035),
                alloy
            );
            spoke.position.set(x, wheelY, z + Math.sign(z) * 0.153);
            spoke.rotation.z = spokeIndex * Math.PI / 5;
            car.add(spoke);
        }
    }
}

function detailBox(size, position, material) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
    mesh.position.set(...position);
    mesh.castShadow = true;
    car.add(mesh);
    return mesh;
}

detailBox([L * 0.24, 0.07, W * 0.84], [L * 0.36, shoulder + 0.04, 0], paint);
detailBox([L * 0.72, 0.08, 0.09], [0, wheelY * 0.68, -W * 0.515], dark);
detailBox([L * 0.72, 0.08, 0.09], [0, wheelY * 0.68, W * 0.515], dark);
detailBox([0.09, 0.31, W * 0.54], [L * 0.505, shoulder * 0.78, 0], dark);
detailBox([0.11, 0.19, W * 0.25], [L * 0.512, shoulder * 1.02, -W * 0.31], headlight);
detailBox([0.11, 0.19, W * 0.25], [L * 0.512, shoulder * 1.02, W * 0.31], headlight);
detailBox([0.11, 0.25, W * 0.22], [-L * 0.505, shoulder * 0.94, -W * 0.34], taillight);
detailBox([0.11, 0.25, W * 0.22], [-L * 0.505, shoulder * 0.94, W * 0.34], taillight);
detailBox([0.13, 0.13, W * 0.34], [L * 0.521, shoulder * 0.62, 0], alloy);
detailBox([0.14, 0.12, W * 0.32], [-L * 0.515, shoulder * 0.63, 0], dark);
detailBox([0.28, 0.13, 0.25], [cabinEnd - L * 0.04, shoulder + 0.35, -W * 0.57], paint);
detailBox([0.28, 0.13, 0.25], [cabinEnd - L * 0.04, shoulder + 0.35, W * 0.57], paint);

for (const z of [-W * 0.503, W * 0.503]) {
    const pillarHeight = Math.max(0.42, cabinRoof - shoulder - 0.12);
    detailBox([0.065, pillarHeight, 0.04], [cabinStart + L * 0.11, shoulder + pillarHeight / 2, z], paint);
    detailBox([0.065, pillarHeight, 0.04], [-L * 0.02, shoulder + pillarHeight / 2, z], paint);
    detailBox([0.065, pillarHeight, 0.04], [cabinEnd - L * 0.11, shoulder + pillarHeight / 2, z], paint);
    detailBox([(cabinEnd - cabinStart) * 0.88, 0.07, 0.045], [(cabinStart + cabinEnd) / 2, shoulder + 0.08, z], paint);
    detailBox([0.20, 0.045, 0.055], [-L * 0.11, shoulder - 0.03, z], alloy);
    detailBox([0.20, 0.045, 0.055], [ L * 0.13, shoulder - 0.03, z], alloy);
}

if (["suv", "mpv", "wagon", "van"].includes(config.bodyStyle)) {
    detailBox([(cabinEnd - cabinStart) * 0.70, 0.055, 0.065], [(cabinStart + cabinEnd) / 2, cabinRoof + 0.09, -W * 0.31], dark);
    detailBox([(cabinEnd - cabinStart) * 0.70, 0.055, 0.065], [(cabinStart + cabinEnd) / 2, cabinRoof + 0.09, W * 0.31], dark);
}

for (const z of [-W * 0.22, -W * 0.11, 0, W * 0.11, W * 0.22]) {
    detailBox([0.045, 0.20, 0.025], [L * 0.523, shoulder * 0.80, z], alloy);
}

const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(L * 1.12, W * 1.55),
    new THREE.MeshBasicMaterial({color: 0x24312f, transparent: true, opacity: 0.22, depthWrite: false})
);
shadow.rotation.x = -Math.PI / 2;
shadow.position.y = 0.012;
scene.add(shadow);

let dragging = false;
let previousX = 0;
renderer.domElement.addEventListener("pointerdown", (event) => {
    dragging = true;
    previousX = event.clientX;
    renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    car.rotation.y += (event.clientX - previousX) * 0.009;
    previousX = event.clientX;
});
renderer.domElement.addEventListener("pointerup", (event) => {
    dragging = false;
    renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener("pointercancel", () => { dragging = false; });

function resize() {
    const width = Math.max(1, view.clientWidth);
    const height = Math.max(1, view.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(view);
resize();
status.classList.add("is-hidden");

let previousTime = performance.now();
function animate(now) {
    const delta = Math.min((now - previousTime) / 1000, 0.05);
    previousTime = now;
    if (!dragging && !prefersReducedMotion) car.rotation.y += delta * 0.08;
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""
)
