const elHostIp = document.getElementById('host-ip');
if (elHostIp) elHostIp.innerText = window.location.hostname || '127.0.0.1 (Local Preview)';
const statusDiv = document.getElementById('status') || { innerText: '', classList: { add: () => { }, remove: () => { } } };
const btnEnterVR = document.getElementById('btn-enter-vr');
const btnExitVR = document.getElementById('btn-exit-vr');
const vrOverlay = document.getElementById('vr-overlay');

// Element bindings for URSim UI
const elValSource = document.getElementById('val-source');
const elValTrigger = document.getElementById('val-trigger');
const elValGrip = document.getElementById('val-grip');
const elValGripperBtn = document.getElementById('val-gripper-btn');
const elValX = document.getElementById('val-x');
const elValY = document.getElementById('val-y');
const elValZ = document.getElementById('val-z');
const elValQx = document.getElementById('val-qx');
const elValQy = document.getElementById('val-qy');
const elValQz = document.getElementById('val-qz');
const elValQw = document.getElementById('val-qw');
const elValMode = document.getElementById('val-mode');

// PolyScope Robot Parameters DOM bindings
const elValTcpX = document.getElementById('val-tcp-x');
const elValTcpY = document.getElementById('val-tcp-y');
const elValTcpZ = document.getElementById('val-tcp-z');
const elValTcpRx = document.getElementById('val-tcp-rx');
const elValTcpRy = document.getElementById('val-tcp-ry');
const elValTcpRz = document.getElementById('val-tcp-rz');

const jointIds = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3'];
const elJointDeg = jointIds.map(j => document.getElementById(`val-${j}-deg`));
const elJointRad = jointIds.map(j => document.getElementById(`val-${j}-rad`));
const elJointPtr = jointIds.map(j => document.getElementById(`ptr-${j}`));
const elValGripperState = document.getElementById('val-gripper-state');

// Safe Joint Limits in Degrees (matching UR3E_LIMITS in pyroki_solver.py)
const JOINT_LIMITS_DEG = [
    [-180.0, 180.0],              // Base
    [-151.8, 11.4],               // Shoulder (-2.65 rad to +0.20 rad)
    [2.9, 171.8],                 // Elbow (+0.05 rad to +3.00 rad)
    [-180.0, 180.0],              // Wrist 1
    [-180.0, 180.0],              // Wrist 2
    [-180.0, 180.0]               // Wrist 3
];

// Anti-Fight Lock state (preventing telemetry from fighting user cockpit input)
let activeCockpitJoint = -1;
let lastCockpitActionTime = 0;
let isDraggingTrack = false;
let dragJointIdx = -1;

function degToPercent(deg) {
    // Thang đo UR [-360 deg, +360 deg] -> range [0%, 100%]
    return Math.max(0, Math.min(100, ((deg + 360.0) / 720.0) * 100.0));
}

function percentToDeg(pct, jointIdx) {
    let rawDeg = (pct / 100.0) * 720.0 - 360.0;
    let limits = JOINT_LIMITS_DEG[jointIdx] || [-180, 180];
    return Math.max(limits[0], Math.min(limits[1], rawDeg));
}

function sendJointCmd(jointIdx, degVal) {
    activeCockpitJoint = jointIdx;
    lastCockpitActionTime = Date.now();
    let radVal = degVal * (Math.PI / 180.0);

    const payload = JSON.stringify({
        type: "joint_cmd",
        joint: jointIdx,
        deg: degVal,
        rad: radVal
    });

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(payload);
    } else {
        fetch(`/api/joint_cmd?joint=${jointIdx}&deg=${degVal.toFixed(2)}`, { method: "GET", cache: "no-store" }).catch(() => { });
    }
}

let ws = null;
let xrSession = null;
let xrRefSpace = null;
let glContext = null;
let activeHandedness = 'right';
let lastDriveTogglePressed = false;
let gripperClosedState = false;
let lastGripperRawPressed = false;
let lastRecordRawPressed = false;
let recordingActiveState = false;

function updateCamDisplay(src) {
    const camImg = document.getElementById('wrist-cam-img');
    const placeholder = document.getElementById('cam-placeholder');
    if (camImg) {
        camImg.src = src;
        camImg.style.display = 'block';
    }
    if (placeholder) {
        placeholder.style.display = 'none';
    }
}

let isWsActive = false;
let lastTelemetryTime = 0;

function updateRobotStatusBadge(text, colorClass) {
    const badge = document.getElementById('robot-status-badge');
    if (badge) {
        if (text) badge.innerText = text;
        if (colorClass) {
            badge.className = `badge ${colorClass}`;
        }
    }
}

function quatToRotVec(w, x, y, z) {
    let angle = 2 * Math.acos(Math.max(-1, Math.min(1, w)));
    let s = Math.sqrt(Math.max(0, 1 - w * w));
    if (s < 0.0001 || angle < 0.0001) return [0, 0, 0];
    return [(x / s) * angle, (y / s) * angle, (z / s) * angle];
}

function updateTelemetryUI(data) {
    if (!data) return;

    // 1. Cập nhật Tool Position (X, Y, Z in mm và RX, RY, RZ in rad)
    if (data.tcp_mm && data.tcp_mm.length >= 3) {
        if (elValTcpX) elValTcpX.innerText = data.tcp_mm[0].toFixed(2);
        if (elValTcpY) elValTcpY.innerText = data.tcp_mm[1].toFixed(2);
        if (elValTcpZ) elValTcpZ.innerText = data.tcp_mm[2].toFixed(2);
    } else if (data.pos && data.pos.length >= 3) {
        if (elValTcpX) elValTcpX.innerText = (data.pos[0] * 1000.0).toFixed(2);
        if (elValTcpY) elValTcpY.innerText = (data.pos[1] * 1000.0).toFixed(2);
        if (elValTcpZ) elValTcpZ.innerText = (data.pos[2] * 1000.0).toFixed(2);
    }

    if (data.rotvec && data.rotvec.length >= 3) {
        if (elValTcpRx) elValTcpRx.innerText = data.rotvec[0].toFixed(3);
        if (elValTcpRy) elValTcpRy.innerText = data.rotvec[1].toFixed(3);
        if (elValTcpRz) elValTcpRz.innerText = data.rotvec[2].toFixed(3);
    } else if (data.quat && data.quat.length >= 4) {
        let rotv = quatToRotVec(data.quat[0], data.quat[1], data.quat[2], data.quat[3]);
        if (elValTcpRx) elValTcpRx.innerText = rotv[0].toFixed(3);
        if (elValTcpRy) elValTcpRy.innerText = rotv[1].toFixed(3);
        if (elValTcpRz) elValTcpRz.innerText = rotv[2].toFixed(3);
    }

    // Fallback backward compatibility for legacy pose elements if present
    if (data.pos && data.pos.length >= 3) {
        if (elValX) elValX.innerText = data.pos[0].toFixed(3);
        if (elValY) elValY.innerText = data.pos[1].toFixed(3);
        if (elValZ) elValZ.innerText = data.pos[2].toFixed(3);
    }
    if (data.quat && data.quat.length >= 4) {
        if (elValQw) elValQw.innerText = data.quat[0].toFixed(3);
        if (elValQx) elValQx.innerText = data.quat[1].toFixed(3);
        if (elValQy) elValQy.innerText = data.quat[2].toFixed(3);
        if (elValQz) elValQz.innerText = data.quat[3].toFixed(3);
    }

    // 2. Cập nhật Joint Position (6 joints: Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3)
    if (data.joints && data.joints.length >= 6) {
        const now = Date.now();
        const isCockpitLocked = (now - lastCockpitActionTime < 450);
        for (let i = 0; i < 6; i++) {
            // Chống xung đột (Anti-Fight Lock): Không ghi đè giá trị nếu người dùng đang chỉnh sửa hoặc vừa tương tác
            let isEditingThis = (document.activeElement === elJointDeg[i]);
            let isLockedThis = isEditingThis || (isCockpitLocked && (activeCockpitJoint === i || activeCockpitJoint === -1));
            if (isLockedThis) {
                continue;
            }

            let rad = data.joints[i];
            let deg = rad * (180.0 / Math.PI);
            if (elJointDeg[i]) elJointDeg[i].innerText = deg.toFixed(2);
            if (elJointRad[i]) elJointRad[i].innerText = `(${rad.toFixed(2)} rad)`;
            if (elJointPtr[i]) {
                // Thang đo UR [-360 deg, +360 deg] -> range [0%, 100%]
                let pct = Math.max(0, Math.min(100, ((deg + 360.0) / 720.0) * 100.0));
                elJointPtr[i].style.left = `${pct.toFixed(1)}%`;
            }
        }
    }

    // 3. Cập nhật Gripper State
    let isGripperClosed = false;
    if (typeof data.gripper !== 'undefined') {
        isGripperClosed = Boolean(data.gripper);
    } else if (data.controller && typeof data.controller.gripper_btn !== 'undefined') {
        isGripperClosed = Boolean(data.controller.gripper_btn);
    }
    gripperClosedState = isGripperClosed;
    if (elValGripperState) {
        elValGripperState.innerText = isGripperClosed ? "Closed (Suction ON)" : "Open (Suction OFF)";
        elValGripperState.className = isGripperClosed ? "badge red" : "badge gray";
    }
    if (elValGripperBtn) {
        elValGripperBtn.innerText = isGripperClosed ? "Closed (Suction ON)" : "Open (Suction OFF)";
        elValGripperBtn.className = isGripperClosed ? "badge red" : "badge gray";
    }

    // 3b. Cập nhật Recording State từ robot telemetry
    if (typeof data.recording !== 'undefined') {
        recordingActiveState = Boolean(data.recording);
    }

    // 4. Cập nhật Robot Status Badge
    let statusText = "Sẵn sàng";
    let colorClass = "green";
    if (data.controller && typeof data.controller.trigger !== 'undefined') {
        statusText = data.controller.trigger ? "Tracking" : "Sẵn sàng";
    } else if (data.status && data.status !== "OK") {
        statusText = data.status;
    } else if (isWsActive) {
        statusText = "Sẵn sàng";
    } else {
        statusText = "Connecting...";
        colorClass = "orange";
    }
    let st = statusText.toLowerCase();
    if (st.includes('lỗi') || st.includes('error') || st.includes('stop') || st.includes('kẹt') || st.includes('disconnected')) {
        colorClass = 'red';
    } else if (st.includes('connecting') || st.includes('reconnecting') || st.includes('waiting')) {
        colorClass = 'orange';
    } else {
        colorClass = 'green';
    }
    updateRobotStatusBadge(statusText, colorClass);

    // 5. Cập nhật Scale Badge nếu có
    if (typeof data.scale !== 'undefined') {
        const elScale = document.getElementById('scale-badge');
        if (elScale) {
            elScale.innerText = `Scale: ${Number(data.scale).toFixed(2)}x`;
        }
    }
}

async function fetchTelemetryPoll() {
    // Nếu WebSocket đang nhận dữ liệu trực tiếp trong 400ms gần nhất thì không cần poll
    if (isWsActive && (Date.now() - lastTelemetryTime < 400)) {
        return;
    }
    try {
        const res = await fetch(`/telemetry.json?t=${Date.now()}`, { cache: "no-store" });
        if (res.ok) {
            const data = await res.json();
            updateTelemetryUI(data);
        }
    } catch (err) {
        // Ignore network glitch
    }
}

// Khởi động vòng lặp polling 10Hz (mỗi 100ms) để cập nhật HUD màn hình PC lập tức
setInterval(fetchTelemetryPoll, 100);

let reconnectDelay = 150; // Khởi đầu retry nhanh 150ms thay vì 2000ms

function connectWebSocket() {
    // 1. Chuẩn hóa hostname để tránh lỗi phân giải IPv6 ::1 gây delay trên Linux
    let rawHost = window.location.hostname;
    let isLoopback = (!rawHost || rawHost === 'localhost' || rawHost === '127.0.0.1');
    let host = isLoopback ? '127.0.0.1' : rawHost;
    let isHttps = window.location.protocol === 'https:';

    // 2. Tối ưu cổng kết nối:
    // Loopback trên PC dùng ws://127.0.0.1:8445 (kết nối siêu nhanh ~2ms, bỏ qua TLS handshake & SSL warning)
    // Kính Quest từ xa qua HTTPS LAN dùng wss://${host}:8444
    const wsUrl = (isHttps && !isLoopback) ? `wss://${host}:8444` : `ws://${host}:8445`;

    try {
        if (ws) {
            ws.onclose = null;
            ws.onerror = null;
            ws.close();
        }
        ws = new WebSocket(wsUrl);
    } catch (err) {
        console.error("Lỗi khởi tạo WebSocket:", err);
        return;
    }

    ws.onopen = () => {
        isWsActive = true;
        reconnectDelay = 150;
        statusDiv.innerText = isHttps ? "Connected (Secure WSS)" : "Connected (Local WS)";
        statusDiv.classList.remove("warning");
        statusDiv.classList.add("success");
        updateRobotStatusBadge("Connected", "green");
        checkWebXR();
        if (xrSession && xrRefSpace) {
            try {
                ws.send(JSON.stringify({
                    type: "session_start",
                    t: Date.now() / 1000.0
                }));
            } catch (e) { }
        }
    };

    ws.onerror = (err) => {
        isWsActive = false;
        statusDiv.innerText = "WS Reconnecting (HTTP Active)";
        updateRobotStatusBadge("Connecting...", "orange");
        console.warn("WS Error (falling back to HTTP Telemetry)", err);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'reset_session') {
                console.warn("[WebXR] Nhận lệnh reset_session từ Server -> Dọn dẹp session sạch");
                disposeSession();
            } else if (data.type === 'haptic' && xrSession) {
                for (let source of xrSession.inputSources) {
                    if (source.handedness === activeHandedness && source.gamepad && source.gamepad.hapticActuators && source.gamepad.hapticActuators.length > 0) {
                        source.gamepad.hapticActuators[0].pulse(data.intensity, data.duration);
                    }
                }
            } else if (data.type === 'teleop_state') {
                setDriveMode(Boolean(data.enabled), false);
                updateRobotStatusBadge(data.enabled ? "Tracking" : "Sẵn sàng (Nhả cò)", "green");
            } else if (data.type === 'camera' && data.image) {
                updateCamDisplay('data:image/jpeg;base64,' + data.image);
            } else if (data.type === 'robot_telemetry') {
                isWsActive = true;
                lastTelemetryTime = Date.now();
                updateTelemetryUI(data);
            }
        } catch (e) { console.error("WS error", e); }
    };

    ws.onclose = () => {
        isWsActive = false;
        statusDiv.innerText = "WS Disconnected (HTTP Active)";
        updateRobotStatusBadge("Connecting...", "orange");
        setTimeout(connectWebSocket, reconnectDelay);
        reconnectDelay = Math.min(2000, reconnectDelay * 1.5);
    };
}

function checkWebXR() {
    if (navigator.xr) {
        navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
            if (supported) {
                btnEnterVR.style.display = 'flex';
                btnEnterVR.innerText = 'Enter VR';
                btnEnterVR.isVR = true;
            } else {
                btnEnterVR.style.display = 'none';
                btnEnterVR.isVR = false;
            }
        }).catch(() => {
            btnEnterVR.style.display = 'none';
            btnEnterVR.isVR = false;
        });
    } else {
        btnEnterVR.style.display = 'none';
        btnEnterVR.isVR = false;
    }
}

async function disposeSession() {
    console.log("[WebXR] Đang dọn dẹp và giải phóng WebXR session...");
    if (xrSession) {
        try {
            await xrSession.end();
        } catch (err) {
            console.warn("[WebXR] Ngoại lệ khi gọi xrSession.end():", err);
        }
        xrSession = null;
    }
    glContext = null;
    xrRefSpace = null;
    btnEnterVR.style.display = 'flex';
    btnExitVR.style.display = 'none';
    vrOverlay.style.display = 'flex';
    statusDiv.innerText = "VR Session Cleaned. Ready to Enter.";
    updateRobotStatusBadge("Sẵn sàng", "green");
}

btnExitVR.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log("[WebXR] Người dùng nhấn Exit VR -> Dọn dẹp session sạch");
    await disposeSession();
});

// Quản lý trạng thái: Khóa cố định Drive Mode để điều khiển Robot thông suốt
let uiLocked = true;
const cockpitPanel = document.getElementById('cockpit-panel');
const overlay = document.getElementById('vr-overlay');
if (cockpitPanel) cockpitPanel.classList.add('drive-mode');
if (overlay) overlay.classList.add('drive-mode');

function setDriveMode(locked, notifyServer = false) {
    uiLocked = true;
    if (cockpitPanel) cockpitPanel.classList.add('drive-mode');
    if (overlay) overlay.classList.add('drive-mode');
    if (notifyServer) {
        fetch(`/api/drive_mode?state=true`).catch(() => { });
    }
}

btnEnterVR.addEventListener('click', async () => {
    if (btnEnterVR.isVR) {
        // Nếu có session cũ còn sót lại (zombie session), ép hủy sạch trước khi tạo mới
        if (xrSession) {
            await disposeSession();
        }

        // DOM Overlay BẮT BUỘC phải hiển thị TRƯỚC KHI requestSession
        vrOverlay.style.display = 'flex';
        statusDiv.innerText = "Entering VR...";

        const sessionInit = {
            optionalFeatures: ['local-floor', 'bounded-floor', 'dom-overlay'],
            domOverlay: { root: vrOverlay }
        };

        navigator.xr.requestSession('immersive-ar', sessionInit)
            .then(onSessionStarted)
            .catch((err) => {
                console.log("AR failed, fallback to VR", err);
                navigator.xr.requestSession('immersive-vr', sessionInit)
                    .then(onSessionStarted)
                    .catch((err2) => {
                        console.error("[WebXR] Lỗi khởi tạo VR session:", err2);
                        statusDiv.innerText = "Enter VR Failed";
                        disposeSession();
                    });
            });
    }
});

function onSessionStarted(session) {
    xrSession = session;
    btnEnterVR.style.display = 'none';
    btnExitVR.style.display = 'flex';
    vrOverlay.style.display = 'flex'; // Hiện Overlay bằng flex để căn giữa
    setDriveMode(true, true); // Luôn mặc định mở Drive Mode ngay khi vào VR để lái robot tức thì
    statusDiv.innerText = "In VR Session";

    session.addEventListener('end', onSessionEnded);

    // Xử lý khi kính tạm thời chuyển trạng thái hiển thị
    session.addEventListener('visibilitychange', () => {
        console.log(`[WebXR] Session visibilityState: ${session.visibilityState}`);
        if (session.visibilityState === 'hidden') {
            // Cảm biến tiệm cận kính hoặc menu che tạm thời: giữ phiên VR, gửi heartbeat giữ an toàn robot
            if (ws && ws.readyState === WebSocket.OPEN) {
                try {
                    ws.send(JSON.stringify({
                        t: Date.now() / 1000.0,
                        tracking_lost: true,
                        trigger: false,
                        grip: false,
                        drive_mode: true
                    }));
                } catch (e) { }
            }
        }
    });

    // Thiết lập WebGL dummy để WebXR chịu chạy loop
    const canvas = document.getElementById('webgl-canvas');
    glContext = canvas.getContext('webgl', { xrCompatible: true, alpha: true });
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glContext) });

    session.requestReferenceSpace('local').then((refSpace) => {
        xrRefSpace = refSpace;
        // Báo cho Server biết có phiên WebXR mới bắt đầu để tự động re-sync mà không cần toggle Teleop
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify({
                    type: "session_start",
                    t: Date.now() / 1000.0
                }));
            } catch (e) { }
        }
        session.requestAnimationFrame(onXRFrame);
    }).catch((refErr) => {
        console.error("[WebXR] Lỗi requestReferenceSpace:", refErr);
        disposeSession();
    });
}

function onSessionEnded(event) {
    console.log("[WebXR] Sự kiện onSessionEnded đã kích hoạt.");
    xrSession = null;
    glContext = null;
    xrRefSpace = null;
    btnEnterVR.style.display = 'flex';
    btnExitVR.style.display = 'none';
    vrOverlay.style.display = 'flex'; // Vẫn duy trì hiển thị Cockpit 2D
    statusDiv.innerText = "VR Session Ended.";
}

function onXRFrame(time, frame) {
    let session = frame.session;
    // Luôn gọi requestAnimationFrame đầu tiên để đảm bảo vòng lặp WebXR không bao giờ bị dừng
    session.requestAnimationFrame(onXRFrame);

    try {
        // Xóa buffer WebGL để không cản trở lớp AR (Xuyên thấu)
        if (glContext) {
            try {
                glContext.bindFramebuffer(glContext.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
                glContext.clearColor(0, 0, 0, 0); // Trong suốt hoàn toàn
                glContext.clear(glContext.COLOR_BUFFER_BIT | glContext.DEPTH_BUFFER_BIT);
            } catch (glErr) {
                // Bỏ qua lỗi WebGL glitch
            }
        }

        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        let sentData = false;
        let activeSource = null;

        if (session && session.inputSources) {
            // Ưu tiên tay phải
            for (let source of session.inputSources) {
                if (!source) continue;
                if (source.targetRayMode === 'gaze') continue;
                if (!source.gripSpace && !source.targetRaySpace) continue;
                if (source.handedness === 'right') {
                    activeSource = source;
                    break;
                }
            }
            // Nếu không có tay phải thì lấy tay trái
            if (!activeSource) {
                for (let source of session.inputSources) {
                    if (!source) continue;
                    if (source.targetRayMode === 'gaze') continue;
                    if (!source.gripSpace && !source.targetRaySpace) continue;
                    if (source.handedness === 'left') {
                        activeSource = source;
                        break;
                    }
                }
            }
        }

        let pose = null;
        if (activeSource && xrRefSpace) {
            activeHandedness = activeSource.handedness;
            let space = activeSource.gripSpace || activeSource.targetRaySpace;
            if (space) {
                try {
                    pose = frame.getPose(space, xrRefSpace);
                } catch (pErr) {
                    pose = null;
                }
            }
        }

        if (activeSource && pose && pose.transform && pose.transform.position && pose.transform.orientation) {
            let triggerPressed = false;
            let gripPressed = false;
            let thumbstickClickPressed = false;
            let recordRawPressed = false;
            let gripperRawPressed = false;
            let gripperBtnPressed = false;
            let btnBPressed = false;

            if (activeSource.gamepad && activeSource.gamepad.buttons) {
                let btns = activeSource.gamepad.buttons;
                let h = activeSource.handedness;
                // Button 0: Index Trigger (bóp trỏ - tịnh tiến, ngưỡng > 0.25)
                if (btns.length > 0) {
                    let b0 = btns[0];
                    let trigVal = (typeof b0.value === 'number') ? b0.value : (b0.pressed ? 1.0 : 0.0);
                    triggerPressed = trigVal > 0.25;
                }
                // Button 1: Hand Grip (bóp hông - xoay, ngưỡng > 0.25)
                if (btns.length > 1) {
                    let b1 = btns[1];
                    let gripVal = (typeof b1.value === 'number') ? b1.value : (b1.pressed ? 1.0 : 0.0);
                    gripPressed = gripVal > 0.25;
                }
                // Button 3: Thumbstick Click (nhấn cần gạt Analog) -> Lệnh Go Home
                if (btns.length > 3) {
                    let b3 = btns[3];
                    thumbstickClickPressed = Boolean(b3.pressed || (typeof b3.value === 'number' && b3.value > 0.5));
                }
                // Button 4: Nút A (tay phải) cho Record Toggle | Nút X (tay trái) cho Gripper
                if (btns.length > 4) {
                    let b4 = btns[4];
                    let b4_val = Boolean(b4.pressed || (typeof b4.value === 'number' && b4.value > 0.5));
                    if (h === 'left') {
                        gripperRawPressed = b4_val;
                    } else {
                        recordRawPressed = b4_val;
                    }
                }
                // Button 5: Nút B (tay phải) hoặc Nút Y (tay trái) - Bật/Tắt Teleop (Grip + B để Hiệu chuẩn bàn máy)
                if (btns.length > 5) {
                    let b5 = btns[5];
                    btnBPressed = Boolean(b5.pressed || (typeof b5.value === 'number' && b5.value > 0.5));
                }
            }

            // Kiểm tra cả tay cầm phụ nếu có (ví dụ tay trái nhấn L3, nút X, hoặc nút Y)
            if (session.inputSources) {
                for (let s of session.inputSources) {
                    if (s && s !== activeSource && s.gamepad && s.gamepad.buttons) {
                        let btns_sub = s.gamepad.buttons;
                        let h_sub = s.handedness;
                        if (btns_sub.length > 3) {
                            let b3_sub = btns_sub[3];
                            if (b3_sub.pressed || (typeof b3_sub.value === 'number' && b3_sub.value > 0.5)) {
                                thumbstickClickPressed = true;
                            }
                        }
                        if (btns_sub.length > 4) {
                            let b4_sub = btns_sub[4];
                            if (b4_sub.pressed || (typeof b4_sub.value === 'number' && b4_sub.value > 0.5)) {
                                if (h_sub === 'left') {
                                    gripperRawPressed = true;
                                } else {
                                    recordRawPressed = true;
                                }
                            }
                        }
                        if (btns_sub.length > 5) {
                            let b5_sub = btns_sub[5];
                            if (b5_sub.pressed || (typeof b5_sub.value === 'number' && b5_sub.value > 0.5)) {
                                btnBPressed = true;
                            }
                        }
                    }
                }
            }

            lastRecordRawPressed = recordRawPressed;
            lastGripperRawPressed = gripperRawPressed;
            gripperBtnPressed = gripperClosedState;

            let pos = pose.transform.position;
            let quat = pose.transform.orientation;

            let headQuat = null;
            let headPos = null;
            let viewerPose = null;
            try {
                viewerPose = frame.getViewerPose(xrRefSpace);
            } catch (vErr) {
                viewerPose = null;
            }
            if (viewerPose && viewerPose.transform) {
                let hp = viewerPose.transform.position;
                let ho = viewerPose.transform.orientation;
                headPos = [hp.x, hp.y, hp.z];
                headQuat = [ho.w, ho.x, ho.y, ho.z];
            }

            let stickTilted = false;
            let stickX = 0.0;
            let stickY = 0.0;
            if (activeSource.gamepad && activeSource.gamepad.axes && activeSource.gamepad.axes.length >= 2) {
                // Chuẩn XR-standard (Meta Quest Touch Controller): axes[0] là X, axes[1] là Y
                stickX = activeSource.gamepad.axes[0];
                stickY = activeSource.gamepad.axes[1];
                // Fallback nếu trục điều khiển nằm ở axes[2], axes[3]
                if (Math.abs(stickX) < 0.05 && Math.abs(stickY) < 0.05 && activeSource.gamepad.axes.length >= 4) {
                    if (Math.abs(activeSource.gamepad.axes[2]) > 0.05 || Math.abs(activeSource.gamepad.axes[3]) > 0.05) {
                        stickX = activeSource.gamepad.axes[2];
                        stickY = activeSource.gamepad.axes[3];
                    }
                }
                stickTilted = Math.abs(stickX) > 0.15 || Math.abs(stickY) > 0.15;
            }

            ws.send(JSON.stringify({
                t: Date.now() / 1000.0,
                hand: activeSource.handedness || "unknown",
                pos: [pos.x, pos.y, pos.z],
                quat: [quat.w, quat.x, quat.y, quat.z],
                head_pos: headPos,
                head_quat: headQuat,
                trigger: triggerPressed,
                grip: gripPressed,
                gripper_raw: gripperRawPressed,
                gripper_toggle_btn: false,
                gripper_btn: gripperBtnPressed,
                button_a: recordRawPressed,
                record_btn: recordRawPressed,
                record_toggle_btn: false,
                button_x: gripperRawPressed,
                button_b: btnBPressed,
                thumbstick: stickTilted,
                thumbstick_click: thumbstickClickPressed,
                thumbstick_axes: [stickX, stickY],
                joystick_tilt: stickTilted,
                home: thumbstickClickPressed,
                drive_mode: true
            }));

            let handName = activeSource.handedness === 'right' ? "Right Hand" : (activeSource.handedness === 'left' ? "Left Hand" : "Unknown");

            // Cập nhật dữ liệu nếu các phần tử cũ còn tồn tại
            if (elValSource) {
                elValSource.innerText = handName;
                elValSource.style.color = "#10b981";
                elValSource.style.fontWeight = "bold";
            }
            if (elValTrigger) {
                elValTrigger.innerText = triggerPressed ? "Pressed" : "Released";
                elValTrigger.className = triggerPressed ? "badge red" : "badge gray";
            }
            updateRobotStatusBadge(triggerPressed ? "Tracking" : "Sẵn sàng (Nhả cò)", "green");
            if (elValGrip) {
                elValGrip.innerText = gripPressed ? "Pressed" : "Released";
                elValGrip.className = gripPressed ? "badge red" : "badge gray";
            }
            if (elValGripperBtn) {
                elValGripperBtn.innerText = gripperBtnPressed ? "Closed (Suction ON)" : "Open (Suction OFF)";
                elValGripperBtn.className = gripperBtnPressed ? "badge red" : "badge gray";
            }

            sentData = true;
        }

        // Khi mất tracking tạm thời hoặc pose == null (tay cầm khuất tầm nhìn kính):
        // Giữ nguyên phiên WebXR, gửi heartbeat an toàn để robot đứng yên mà không ngắt session
        if (!sentData) {
            try {
                ws.send(JSON.stringify({
                    t: Date.now() / 1000.0,
                    tracking_lost: true,
                    trigger: false,
                    grip: false,
                    gripper_raw: false,
                    button_b: false,
                    drive_mode: true
                }));
            } catch (eSend) { }

            if (elValSource) {
                elValSource.innerText = "Tracking Lost (Holding)";
                elValSource.style.color = "#f59e0b";
                elValSource.style.fontWeight = "normal";
            }
            updateRobotStatusBadge("Sẵn sàng (Khuất tay cầm)", "orange");
        }
    } catch (errLoop) {
        console.error("[WebXR] Ngoại lệ bảo vệ trong onXRFrame:", errLoop);
    }
}

// Quản lý luồng stream camera ảo cổ tay
let lastCamUpdate = 0;
let isPolling = false;

function pollCameraFrame() {
    if (Date.now() - lastCamUpdate < 250 || isPolling) return;
    isPolling = true;

    const pollImg = new Image();
    pollImg.onload = () => {
        isPolling = false;
        updateCamDisplay(pollImg.src);
    };
    pollImg.onerror = () => {
        isPolling = false;
        // Nếu đang mở bằng file:/// và chưa nạp được, thử fallback qua cổng HTTPS 8443
        if (window.location.protocol === 'file:' && !pollImg.hasFallenBack) {
            pollImg.hasFallenBack = true;
            let host = window.location.hostname || '127.0.0.1';
            pollImg.src = `https://${host}:8443/wrist_camera.jpg?t=${Date.now()}`;
        }
    };
    // Đường dẫn tương đối không có dấu gạch chéo đầu giúp tương thích hoàn hảo cả file:/// lẫn http/https
    pollImg.src = `wrist_camera.jpg?t=${Date.now()}`;
}

// Chỉ kích hoạt polling khi mở file cục bộ qua file:/// (do không kết nối được stream /video_feed)
if (window.location.protocol === 'file:') {
    setInterval(pollCameraFrame, 200);
}

// =========================================================================
// Interactive Joint-Space Control Handlers (Quest Cockpit)
// =========================================================================
function initCockpitJointControls() {
    function updateFromPointer(e, i, trackEl) {
        const rect = trackEl.getBoundingClientRect();
        if (rect.width <= 0) return;
        let pct = ((e.clientX - rect.left) / rect.width) * 100.0;
        pct = Math.max(0, Math.min(100, pct));
        let deg = percentToDeg(pct, i);
        let rad = deg * (Math.PI / 180.0);
        let clampedPct = degToPercent(deg);

        if (elJointDeg[i]) elJointDeg[i].innerText = deg.toFixed(2);
        if (elJointRad[i]) elJointRad[i].innerText = `(${rad.toFixed(2)} rad)`;
        if (elJointPtr[i]) elJointPtr[i].style.left = `${clampedPct.toFixed(1)}%`;

        activeCockpitJoint = i;
        lastCockpitActionTime = Date.now();
        sendJointCmd(i, deg);
    }

    function setupArrowButtons(i, leftArr, rightArr) {
        function jogOnce(deltaDeg) {
            let curDeg = parseFloat(elJointDeg[i].innerText) || 0.0;
            let limits = JOINT_LIMITS_DEG[i] || [-180, 180];
            let newDeg = Math.max(limits[0], Math.min(limits[1], curDeg + deltaDeg));
            let rad = newDeg * (Math.PI / 180.0);
            let pct = degToPercent(newDeg);

            if (elJointDeg[i]) elJointDeg[i].innerText = newDeg.toFixed(2);
            if (elJointRad[i]) elJointRad[i].innerText = `(${rad.toFixed(2)} rad)`;
            if (elJointPtr[i]) elJointPtr[i].style.left = `${pct.toFixed(1)}%`;

            activeCockpitJoint = i;
            lastCockpitActionTime = Date.now();
            sendJointCmd(i, newDeg);
        }

        let holdTimeout = null;
        let repeatInterval = null;

        function startJog(deltaDeg, el) {
            jogOnce(deltaDeg);
            if (el) el.classList.add('active');
            holdTimeout = setTimeout(() => {
                repeatInterval = setInterval(() => {
                    jogOnce(deltaDeg);
                }, 70);
            }, 220);
        }

        function stopJog(el) {
            if (holdTimeout) { clearTimeout(holdTimeout); holdTimeout = null; }
            if (repeatInterval) { clearInterval(repeatInterval); repeatInterval = null; }
            if (el) el.classList.remove('active');
            activeCockpitJoint = i;
            lastCockpitActionTime = Date.now();
        }

        if (leftArr) {
            leftArr.addEventListener('pointerdown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                startJog(-1.0, leftArr);
            });
            leftArr.addEventListener('pointerup', () => stopJog(leftArr));
            leftArr.addEventListener('pointerleave', () => stopJog(leftArr));
            leftArr.addEventListener('pointercancel', () => stopJog(leftArr));
        }

        if (rightArr) {
            rightArr.addEventListener('pointerdown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                startJog(1.0, rightArr);
            });
            rightArr.addEventListener('pointerup', () => stopJog(rightArr));
            rightArr.addEventListener('pointerleave', () => stopJog(rightArr));
            rightArr.addEventListener('pointercancel', () => stopJog(rightArr));
        }
    }

    function setupDirectDegreeInput(i, valBox, spanDeg) {
        if (!valBox || !spanDeg) return;
        valBox.setAttribute('title', 'Bấm để nhập góc độ (° trực tiếp), Enter để xác nhận');

        function enterEdit() {
            activeCockpitJoint = i;
            lastCockpitActionTime = Date.now() + 60000;
            spanDeg.contentEditable = 'true';
            spanDeg.focus();
            try {
                const range = document.createRange();
                range.selectNodeContents(spanDeg);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            } catch (_) { }
        }

        function commitEdit() {
            spanDeg.contentEditable = 'false';
            let valStr = spanDeg.innerText.trim();
            let num = parseFloat(valStr);
            let limits = JOINT_LIMITS_DEG[i] || [-180, 180];
            if (!isNaN(num)) {
                let clamped = Math.max(limits[0], Math.min(limits[1], num));
                let rad = clamped * (Math.PI / 180.0);
                let pct = degToPercent(clamped);
                spanDeg.innerText = clamped.toFixed(2);
                if (elJointRad[i]) elJointRad[i].innerText = `(${rad.toFixed(2)} rad)`;
                if (elJointPtr[i]) elJointPtr[i].style.left = `${pct.toFixed(1)}%`;
                activeCockpitJoint = i;
                lastCockpitActionTime = Date.now();
                sendJointCmd(i, clamped);
            } else {
                activeCockpitJoint = -1;
                lastCockpitActionTime = 0;
            }
        }

        function cancelEdit() {
            spanDeg.contentEditable = 'false';
            activeCockpitJoint = -1;
            lastCockpitActionTime = 0;
        }

        valBox.addEventListener('click', (e) => {
            e.stopPropagation();
            enterEdit();
        });

        spanDeg.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                commitEdit();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                cancelEdit();
            }
        });

        spanDeg.addEventListener('blur', () => {
            commitEdit();
        });
    }

    for (let i = 0; i < jointIds.length; i++) {
        const itemEl = elJointDeg[i] ? elJointDeg[i].closest('.ur-joint-item') : null;
        if (!itemEl) continue;

        const leftArr = itemEl.querySelector('.ur-arr-left');
        const rightArr = itemEl.querySelector('.ur-arr-right');
        const trackWrap = itemEl.querySelector('.ur-track-wrap');
        const valBox = elJointDeg[i].closest('.ur-joint-val-box');

        // 1. Slider dragging on ruler track
        if (trackWrap) {
            trackWrap.addEventListener('pointerdown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                isDraggingTrack = true;
                dragJointIdx = i;
                trackWrap.classList.add('dragging');
                try {
                    trackWrap.setPointerCapture(e.pointerId);
                } catch (_) { }
                updateFromPointer(e, i, trackWrap);
            });

            trackWrap.addEventListener('pointermove', (e) => {
                if (!isDraggingTrack || dragJointIdx !== i) return;
                updateFromPointer(e, i, trackWrap);
            });

            trackWrap.addEventListener('pointerup', (e) => {
                if (!isDraggingTrack || dragJointIdx !== i) return;
                try {
                    trackWrap.releasePointerCapture(e.pointerId);
                } catch (_) { }
                trackWrap.classList.remove('dragging');
                isDraggingTrack = false;
                dragJointIdx = -1;
                activeCockpitJoint = i;
                lastCockpitActionTime = Date.now();
            });

            trackWrap.addEventListener('pointercancel', (e) => {
                if (!isDraggingTrack || dragJointIdx !== i) return;
                try {
                    trackWrap.releasePointerCapture(e.pointerId);
                } catch (_) { }
                trackWrap.classList.remove('dragging');
                isDraggingTrack = false;
                dragJointIdx = -1;
                activeCockpitJoint = i;
                lastCockpitActionTime = Date.now();
            });
        }

        // 2. Green arrow buttons (click and hold repeat)
        setupArrowButtons(i, leftArr, rightArr);

        // 3. Direct degree input box
        setupDirectDegreeInput(i, valBox, elJointDeg[i]);
    }
}

// Start
connectWebSocket();
initCockpitJointControls();

// Auto-assign the correct IP for Viser iframe based on current host
const viserIframe = document.getElementById('viser-iframe');
if (viserIframe) {
    let host = window.location.hostname || '127.0.0.1';
    viserIframe.src = `http://${host}:8080`;
}
