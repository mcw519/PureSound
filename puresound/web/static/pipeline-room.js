/* Pipeline screen, 3-D room: the bank room a row was rendered in -- walls,
 * obstacles, the microphone and every source, the ones the row used lit in
 * their role's colour, joined to the microphone, and sending wavefronts toward
 * it, with the room's X, Y and Z axes.  three.js is vendored (vendor/three/README.md) and reached through the
 * page's import map; pipeline.js draws a floor plan instead when this module
 * or WebGL is unavailable.  Room coordinates are metres with z up; the scene
 * puts the room's centre at the origin with y up. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BACKGROUND = 0x010120;
const UNUSED = 0x6f7078;
const MIC = 0x6fd3db;
// A slowed speed of sound for the wavefronts, so they can be followed by eye.
const WAVE_SPEED = 1.4;
const WAVE_PERIOD = 1.1;
const AXIS_COLORS = { X: 0xff6b6b, Y: 0x7ee0a1, Z: 0x6fa8ff };
const TICK = 0xa4a6d8;
const MATERIAL_COLORS = { curtain: 0x3b3f6b, wood: 0x6b5238, glass: 0x5a8aa6, concrete: 0x55585f, fabric: 0x4b3f6b };

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function supported() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(window.WebGLRenderingContext && (canvas.getContext("webgl2") || canvas.getContext("webgl")));
  } catch {
    return false;
  }
}

function colour(value) {
  return new THREE.Color(value);
}

function rippleMaterial(tint) {
  return new THREE.ShaderMaterial({
    uniforms: { tint: { value: tint }, strength: { value: 0 } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `varying vec3 surfaceNormal; varying vec3 eye;
      void main() {
        vec4 point = modelViewMatrix * vec4(position, 1.0);
        surfaceNormal = normalize(normalMatrix * normal); eye = -point.xyz;
        gl_Position = projectionMatrix * point;
      }`,
    fragmentShader: `uniform vec3 tint; uniform float strength;
      varying vec3 surfaceNormal; varying vec3 eye;
      void main() {
        float rim = pow(1.0 - abs(dot(normalize(surfaceNormal), normalize(eye))), 3.0);
        gl_FragColor = vec4(tint, strength * (0.025 + rim));
      }`,
  });
}

/* A text label that always faces the camera. */
function label(text, color = "#ffffff", size = 0.16) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const font = "500 44px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.font = font;
  const width = Math.ceil(context.measureText(text).width) + 24;
  canvas.width = width;
  canvas.height = 64;
  context.font = font;
  context.fillStyle = "rgba(1,1,32,0.72)";
  context.fillRect(0, 6, width, 52);
  context.fillStyle = color;
  context.textBaseline = "middle";
  context.fillText(text, 12, 33);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.scale.set((size * width) / 64, size, 1);
  sprite.renderOrder = 10;
  return sprite;
}

class RoomView {
  static supported() {
    return supported();
  }

  constructor(host) {
    this.host = host;
    host.innerHTML = `<div class="pipe-room-canvas"></div><div class="pipe-room-caption">drag to turn · scroll to zoom · right-drag to pan</div><div class="pipe-legend" data-room-legend></div>`;
    this.canvasHost = host.querySelector(".pipe-room-canvas");
    this.legend = host.querySelector("[data-room-legend]");
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(BACKGROUND, 1);
    this.canvasHost.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(BACKGROUND, 12, 30);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener("change", () => this.request());
    this.controls.addEventListener("start", () => { this.moved = true; });
    this.scene.add(new THREE.HemisphereLight(0xcfd3ff, 0x1a1b3a, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(4, 8, 5);
    this.scene.add(sun);
    this.room = new THREE.Group();
    this.scene.add(this.room);
    this.sources = [];
    this.waves = [];
    this.wavesOn = true;
    this.emphasis = [];
    this.visible = true;
    this.frame = 0;
    this.last = performance.now();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvasHost);
    this.visibility = new IntersectionObserver((entries) => {
      this.visible = entries.some((entry) => entry.isIntersecting);
      if (this.visible) this.request();
    });
    this.visibility.observe(this.canvasHost);
    this.onVisibility = () => { if (!document.hidden) this.request(); };
    document.addEventListener("visibilitychange", this.onVisibility);
    this.resize();
  }

  /* room: the report's room, sources carrying display roles and optionally
   * their own colour; options.colors and options.labels map those roles to a
   * colour and a legend name; options.waves turns the emitted wavefront
   * animation off and options.axes the X/Y/Z axes. */
  show(room, { colors = {}, labels = {}, waves = true, axes = true, soft = false } = {}) {
    this.clear();
    this.moved = false;
    this.colors = colors;
    this.labels = labels;
    this.wavesOn = waves;
    this.renderer.setClearColor(soft ? 0x050b1b : BACKGROUND, 1);
    this.scene.fog.color.setHex(soft ? 0x050b1b : BACKGROUND);
    if(labels.controls)this.host.querySelector(".pipe-room-caption").textContent=labels.controls;
    const [width, depth, height] = room.room_dim;
    const place = ([x, y, z]) => new THREE.Vector3(x - width / 2, z, depth / 2 - y);
    this.place = place;

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), new THREE.MeshStandardMaterial({ color: 0x14163a, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    this.room.add(floor);
    const grid = [];
    for (let x = -width / 2; x <= width / 2 + 1e-6; x += 0.5) grid.push(x, 0.002, -depth / 2, x, 0.002, depth / 2);
    for (let z = -depth / 2; z <= depth / 2 + 1e-6; z += 0.5) grid.push(-width / 2, 0.002, z, width / 2, 0.002, z);
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute("position", new THREE.Float32BufferAttribute(grid, 3));
    this.room.add(new THREE.LineSegments(gridGeometry, new THREE.LineBasicMaterial({ color: 0x2b2e5c, transparent: true, opacity: soft ? 0.4 : 1 })));

    const box = new THREE.BoxGeometry(width, height, depth);
    const walls = new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: 0x8f8cff, transparent: true, opacity: 0.04, side: THREE.BackSide, depthWrite: false }));
    walls.position.y = height / 2;
    this.room.add(walls);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0x8f8cff, transparent: true, opacity: soft ? 0.23 : 0.55 }));
    edges.position.y = height / 2;
    this.room.add(edges);
    if (axes) {
      const rulers = this.axes(width, depth, height);
      if (soft) rulers.traverse((part) => { if (part.material) { part.material.transparent = true; part.material.opacity = 0.55; } });
      this.room.add(rulers);
    }

    (room.obstacles || []).forEach((obstacle) => {
      const points = obstacle.footprint || [];
      if (points.length < 3) return;
      const shape = new THREE.Shape(points.map(([x, y]) => new THREE.Vector2(x - width / 2, y - depth / 2)));
      const tall = Math.max(0.02, (obstacle.z_max || 0) - (obstacle.z_min || 0));
      const geometry = new THREE.ExtrudeGeometry(shape, { depth: tall, bevelEnabled: false });
      geometry.rotateX(-Math.PI / 2);
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: MATERIAL_COLORS[obstacle.material] || 0x4a4d78, roughness: 0.8, transparent: true, opacity: 0.82 }));
      mesh.position.y = obstacle.z_min || 0;
      const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 30), new THREE.LineBasicMaterial({ color: 0xa4a6d8, transparent: true, opacity: 0.35 }));
      outline.position.y = obstacle.z_min || 0;
      this.room.add(mesh, outline);
    });

    const receiver = place(room.receiver);
    this.receiver = receiver;
    const mic = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 16), new THREE.MeshStandardMaterial({ color: MIC, emissive: MIC, emissiveIntensity: 0.6 }));
    mic.position.copy(receiver);
    this.room.add(mic, this.stand(receiver, MIC));
    const micLabel = label("mic", "#6fd3db", 0.2);
    micLabel.position.copy(receiver).add(new THREE.Vector3(0, 0.3, 0));
    this.room.add(micLabel);

    (room.sources || []).forEach((source) => {
      if (!Array.isArray(source.position)) return;
      const role = (source.roles || [])[0] || null;
      const tint = role ? colour(source.color || colors[role] || "#ffffff") : colour(UNUSED);
      const position = place(source.position);
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(role ? 0.1 : 0.06, 24, 16),
        role ? new THREE.MeshStandardMaterial({ color: tint, emissive: tint, emissiveIntensity: 0.45 }) : new THREE.MeshBasicMaterial({ color: tint, wireframe: true, transparent: true, opacity: 0.6 }),
      );
      ball.position.copy(position);
      const entry = { source, role, roles: source.roles || [], ball, position, parts: [ball], tint, phase: Math.random() * WAVE_PERIOD };
      this.room.add(ball, this.stand(position, role ? tint : UNUSED, entry));
      if (role) {
        const ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([position, receiver]), new THREE.LineDashedMaterial({ color: tint, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: soft ? 0.35 : 0.9 }));
        ray.computeLineDistances();
        const tag = label(source.dynamic ? source.label : `${source.label} · ${Number(source.distance_m).toFixed(2)} m`, `#${tint.getHexString()}`, 0.2);
        tag.position.copy(position).add(new THREE.Vector3(0, 0.3, 0));
        this.room.add(ray, tag);
        entry.parts.push(ray, tag);
      } else {
        const tag = label(source.label, "#8b8d99", 0.15);
        tag.position.copy(position).add(new THREE.Vector3(0, 0.18, 0));
        this.room.add(tag);
        entry.parts.push(tag);
      }
      if(source.dynamic){
        const yaw=(source.yaw_deg??0)*Math.PI/180,pitch=(source.pitch_deg??0)*Math.PI/180;
        const aim=new THREE.Vector3(Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch),-Math.sin(yaw)*Math.cos(pitch));
        entry.arrow=new THREE.ArrowHelper(aim,position,.35,tint,.1,.05);
        this.room.add(entry.arrow);
      }
      this.sources.push(entry);
    });

    this.renderLegend();
    this.frameCamera(width, depth, height);
    this.applyEmphasis();
    this.request();
  }

  updateSourcePositions(positions, orientations=[]) {
    this.sources.forEach((entry, i) => {
      if (!positions[i]) return;
      const next = this.place(positions[i]);
      const delta = next.clone().sub(entry.position);
      entry.ball.position.copy(next);
      for (const part of entry.parts.slice(1)) {
        if (part.isSprite) part.position.add(delta);
        else if (part.isMesh) part.position.set(next.x, 0.004, next.z);
        else if (part.isLine) {
          const points = part.material.isLineDashedMaterial && part.material.dashSize === 0.12
            ? [next, this.receiver] : [next, new THREE.Vector3(next.x, 0, next.z)];
          part.geometry.setFromPoints(points);
          part.computeLineDistances();
        }
      }
      if(entry.arrow){
        const state=orientations[i]??entry.source,yaw=(state.yaw_deg??0)*Math.PI/180,pitch=(state.pitch_deg??0)*Math.PI/180;
        entry.arrow.position.copy(next);
        entry.arrow.setDirection(new THREE.Vector3(Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch),-Math.sin(yaw)*Math.cos(pitch)));
      }
      entry.position.copy(next);
    });
    this.request();
  }

  /* Clock-driven volume display for the world screen. Fixed mesh count, with
   * origins supplied at each front's birth so fronts do not follow a walker.
   * Static pipeline waves keep their existing independent animation. */
  updateAudioWaves(sources) {
    this.sources.forEach((entry, i) => {
      const state = sources[i] || { level: 0, fronts: [] };
      entry.ball.material.emissiveIntensity = 0.45 + state.level * 1.2;
      if (!entry.audioShells && state.fronts.length && !reducedMotion.matches) {
        entry.audioShells = Array.from({ length: 4 }, () => {
          const shell = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 24), rippleMaterial(entry.tint));
          this.room.add(shell);
          return shell;
        });
      }
      (entry.audioShells || []).forEach((shell, j) => {
        const front = state.fronts[j];
        shell.visible = Boolean(front) && !reducedMotion.matches;
        if (!shell.visible) return;
        shell.position.copy(this.place(front.position));
        shell.scale.setScalar(front.radius);
        shell.material.uniforms.strength.value = front.opacity;
      });
    });
    this.request();
  }

  /* Rulers just outside the room's origin corner along X (width), Y (depth)
   * and Z (height), as on the floor plan: arrows labelled with the room's
   * size, crossed by a tick and a number every metre on the floor. */
  axes(width, depth, height) {
    const group = new THREE.Group();
    const gap = 0.15;
    const origin = this.place([-gap, -gap, 0]);
    const axes = [["X", width, new THREE.Vector3(1, 0, 0)], ["Y", depth, new THREE.Vector3(0, 0, -1)], ["Z", height, new THREE.Vector3(0, 1, 0)]];
    for (const [name, size, direction] of axes) {
      const length = size + gap + 0.5;
      group.add(new THREE.ArrowHelper(direction, origin, length, AXIS_COLORS[name], 0.16, 0.08));
      const tag = label(`${name} · ${+size.toFixed(2)} m`, `#${colour(AXIS_COLORS[name]).getHexString()}`, 0.2);
      tag.position.copy(origin).addScaledVector(direction, length + 0.3);
      group.add(tag);
    }
    const step = Math.max(width, depth) > 14 ? 2 : 1;
    const ticks = [];
    const marks = [];
    for (let x = 0; x <= width + 1e-6; x += step) {
      ticks.push(...this.place([x, -gap + 0.07, 0]).toArray(), ...this.place([x, -gap - 0.07, 0]).toArray());
      marks.push([x, [x, -gap - 0.3, 0]]);
    }
    for (let y = step; y <= depth + 1e-6; y += step) {
      ticks.push(...this.place([-gap + 0.07, y, 0]).toArray(), ...this.place([-gap - 0.07, y, 0]).toArray());
      marks.push([y, [-gap - 0.3, y, 0]]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(ticks, 3));
    group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: TICK })));
    for (const [value, at] of marks) {
      const mark = label(String(value), `#${colour(TICK).getHexString()}`, 0.15);
      mark.position.copy(this.place(at));
      group.add(mark);
    }
    return group;
  }

  /* A dashed drop to the floor with a footprint ring, so height reads in 3-D. */
  stand(position, color, entry = null) {
    const group = new THREE.Group();
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([position, new THREE.Vector3(position.x, 0, position.z)]), new THREE.LineDashedMaterial({ color, dashSize: 0.05, gapSize: 0.05, transparent: true, opacity: 0.45 }));
    line.computeLineDistances();
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.09, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(position.x, 0.004, position.z);
    group.add(line, ring);
    if (entry) entry.parts.push(line, ring);
    return group;
  }

  /* Far enough back that the room's bounding sphere fits the narrower of the
   * two view angles, looking down from a front corner. */
  frameCamera(width, depth, height) {
    const radius = Math.hypot(width, depth, height) / 2;
    const vertical = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * this.camera.aspect);
    const distance = (radius / Math.sin(Math.min(vertical, horizontal))) * 1.02;
    const direction = new THREE.Vector3(0.62, 0.7, 1).normalize();
    this.controls.target.set(0, height * 0.28, 0);
    this.camera.position.copy(this.controls.target).addScaledVector(direction, distance);
    this.camera.near = Math.max(0.05, distance / 100);
    this.camera.far = distance * 4;
    this.camera.updateProjectionMatrix();
    this.scene.fog.near = distance * 0.9;
    this.scene.fog.far = distance * 2.6;
    this.controls.minDistance = radius * 0.4;
    this.controls.maxDistance = distance * 2.5;
    this.controls.update();
    this.framed = { width, depth, height };
  }

  renderLegend() {
    const used = [...new Set(this.sources.map((entry) => entry.role).filter(Boolean))];
    const items = used.map((role) => `<span><i style="background:${this.colors[role] || "#fff"}"></i>${this.labels[role] || role}</span>`);
    if (this.sources.some((entry) => !entry.role)) items.push(`<span><i class="is-unused"></i>${this.labels.unused || "not used on this row"}</span>`);
    items.push(`<span><i class="is-mic"></i>${this.labels.mic || "microphone"}</span>`);
    this.legend.innerHTML = items.join("");
  }

  /* Light the sources whose roles acted in the selected stage; dim the rest. */
  highlight(roles) {
    this.emphasis = roles || [];
    this.applyEmphasis();
    this.request();
  }

  applyEmphasis() {
    const focus = this.emphasis.length > 0;
    this.sources.forEach((entry) => {
      const on = !focus || entry.roles.some((role) => this.emphasis.includes(role));
      entry.parts.forEach((part) => {
        const material = part.material;
        if (!material) return;
        material.transparent = true;
        if (material.userData.baseOpacity === undefined) material.userData.baseOpacity = material.opacity;
        material.opacity = on ? material.userData.baseOpacity : material.userData.baseOpacity * 0.18;
      });
      entry.active = Boolean(entry.role) && on && (!focus || entry.roles.some((role) => this.emphasis.includes(role)));
    });
  }

  spawnWaves(dt) {
    if (!this.wavesOn || reducedMotion.matches || !this.receiver) return;
    this.sources.forEach((entry) => {
      if (!entry.active) return;
      entry.phase += dt;
      if (entry.phase < WAVE_PERIOD) return;
      entry.phase = 0;
      const shell = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 18), new THREE.MeshBasicMaterial({ color: entry.tint, transparent: true, opacity: 0.35, wireframe: true, depthWrite: false }));
      shell.position.copy(entry.position);
      shell.scale.setScalar(0.05);
      shell.userData = { age: 0, reach: entry.position.distanceTo(this.receiver) * 1.15 + 0.2 };
      this.room.add(shell);
      this.waves.push(shell);
    });
  }

  animateWaves(dt) {
    this.waves = this.waves.filter((shell) => {
      shell.userData.age += dt;
      const radius = 0.05 + shell.userData.age * WAVE_SPEED;
      if (radius > shell.userData.reach) {
        this.room.remove(shell);
        shell.geometry.dispose();
        shell.material.dispose();
        return false;
      }
      shell.scale.setScalar(radius);
      shell.material.opacity = 0.35 * (1 - radius / shell.userData.reach);
      return true;
    });
  }

  request() {
    if (this.frame) return;
    this.frame = requestAnimationFrame((now) => this.tick(now));
  }

  tick(now) {
    this.frame = 0;
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    const moving = this.controls.update();
    this.spawnWaves(dt);
    this.animateWaves(dt);
    this.renderer.render(this.scene, this.camera);
    const animating = this.wavesOn && !reducedMotion.matches && this.sources.some((entry) => entry.active);
    if (this.visible && !document.hidden && this.host.isConnected && (moving || animating || this.waves.length)) this.request();
  }

  resize() {
    const width = this.canvasHost.clientWidth || 600;
    const height = this.canvasHost.clientHeight || 380;
    this.renderer.setSize(width, height, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.framed && !this.moved) this.frameCamera(this.framed.width, this.framed.depth, this.framed.height);
    this.request();
  }

  clear() {
    this.waves = [];
    this.sources = [];
    this.room.traverse((object) => {
      object.geometry?.dispose?.();
      object.material?.map?.dispose?.();
      object.material?.dispose?.();
    });
    this.room.clear();
  }

  dispose() {
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.visibility.disconnect();
    this.clear();
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.controls.dispose();
    this.renderer.dispose();
    // Browsers cap live WebGL contexts; toggling the view must not pile them up.
    this.renderer.forceContextLoss();
    this.host.innerHTML = "";
  }
}

window.PureSoundRoomView = RoomView;
window.dispatchEvent(new Event("puresound:room-view-ready"));
