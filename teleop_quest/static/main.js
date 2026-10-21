        document.getElementById('host-ip').innerText = window.location.hostname || '127.0.0.1 (Local Preview)';
        const statusDiv = document.getElementById('status');
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

        let ws = null;
        let xrSession = null;
        let xrRefSpace = null;
        let glContext = null;
        let activeHandedness = 'right';
        let lastDriveTogglePressed = false;
        let gripperClosedState = false;
        let lastGripperRawPressed = false;

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
                for (let i = 0; i < 6; i++) {
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

            // 4. Cập nhật Robot Status Badge
            const elRobotStatus = document.getElementById('robot-status-badge');
            if (elRobotStatus && data.status) {
                elRobotStatus.innerText = data.status;
                let st = data.status.toLowerCase();
                if (st.includes('tracking') || st.includes('home') || st.includes('bật') || st.includes('sẵn sàng') || st.includes('standby')) {
                    elRobotStatus.className = 'badge green';
                } else if (st.includes('lỗi') || st.includes('error') || st.includes('stop') || st.includes('kẹt')) {
                    elRobotStatus.className = 'badge red';
                } else {
                    elRobotStatus.className = 'badge orange';
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

        function connectWebSocket() {
            let host = window.location.hostname || '127.0.0.1';
            let isHttps = window.location.protocol === 'https:';
            // Meta Quest WebXR chạy https: -> dùng wss:// cổng 8444
            // Xem PC (file:/// hoặc http:) -> dùng ws:// cổng 8445 không cần chứng chỉ SSL tự ký
            const wsUrl = isHttps ? `wss://${host}:8444` : `ws://${host}:8445`;
            
            try {
                ws = new WebSocket(wsUrl);
            } catch (err) {
                console.error("Lỗi khởi tạo WebSocket:", err);
                return;
            }

            ws.onopen = () => {
                isWsActive = true;
                statusDiv.innerText = isHttps ? "Connected (Secure WSS)" : "Connected (Local WS)";
                statusDiv.classList.remove("warning");
                statusDiv.classList.add("success");
                checkWebXR();
            };

            ws.onerror = (err) => {
                isWsActive = false;
                statusDiv.innerText = "WS Reconnecting (HTTP Active)";
                console.warn("WS Error (falling back to HTTP Telemetry)", err);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'haptic' && xrSession) {
                        for (let source of xrSession.inputSources) {
                            if (source.handedness === activeHandedness && source.gamepad && source.gamepad.hapticActuators && source.gamepad.hapticActuators.length > 0) {
                                source.gamepad.hapticActuators[0].pulse(data.intensity, data.duration);
                            }
                        }
                    } else if (data.type === 'teleop_state') {
                        setDriveMode(Boolean(data.enabled), false);
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
                setTimeout(connectWebSocket, 2000);
            };
        }

        function checkWebXR() {
            btnEnterVR.style.display = 'flex';
            if (navigator.xr) {
                navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                    if (supported) {
                        btnEnterVR.innerText = 'START IMMERSIVE VR';
                        btnEnterVR.isVR = true;
                    } else {
                        btnEnterVR.innerText = 'START DASHBOARD';
                        btnEnterVR.isVR = false;
                    }
                });
            } else {
                btnEnterVR.innerText = 'START DASHBOARD';
                btnEnterVR.isVR = false;
            }
        }


        
        btnExitVR.addEventListener('click', () => {
            if (xrSession) {
                xrSession.end();
            } else {
                // Exit Preview Mode
                vrOverlay.style.display = 'none';
                document.getElementById('launcher-screen').style.display = 'flex';
                document.body.style.backgroundColor = '#f0f0f0';
            }
        });

        // Quản lý chuyển đổi giữa Edit Mode (chỉnh UI) và Drive Mode (lái Robot)
        let uiLocked = true;
        const btnToggleUI = document.getElementById('btn-toggle-ui');
        const cockpitPanel = document.getElementById('cockpit-panel');

        function setDriveMode(locked, notifyServer = true) {
            uiLocked = Boolean(locked);
            const overlay = document.getElementById('vr-overlay');

            if (uiLocked) {
                if (cockpitPanel) cockpitPanel.classList.add('drive-mode');
                if (overlay) overlay.classList.add('drive-mode');
                btnToggleUI.innerHTML = '<span></span> Unlock UI (Edit Mode) [B/Y]';
                btnToggleUI.style.background = '#10b981';
                if (elValMode) {
                    elValMode.innerText = "Drive Mode";
                    elValMode.className = "badge green";
                }
            } else {
                if (cockpitPanel) cockpitPanel.classList.remove('drive-mode');
                if (overlay) overlay.classList.remove('drive-mode');
                btnToggleUI.innerHTML = '<span></span> Lock UI (Drive Mode) [B/Y]';
                btnToggleUI.style.background = '#f59e0b';
                if (elValMode) {
                    elValMode.innerText = "Edit Mode";
                    elValMode.className = "badge orange";
                }
            }

            // Đồng bộ trạng thái lên Python server
            if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'ui_state',
                    drive_mode: uiLocked
                }));
            } else if (notifyServer) {
                fetch(`/api/drive_mode?state=${uiLocked}`).catch(() => {});
            }
        }

        btnToggleUI.addEventListener('click', (e) => {
            e.stopPropagation();
            setDriveMode(!uiLocked, true);
        });

        const btnHomeVR = document.getElementById('btn-home-vr');
        if (btnHomeVR) {
            btnHomeVR.addEventListener('click', (e) => {
                e.stopPropagation();
                triggerGoHome(activeSource);
            });
        }

        let lastThumbstickPressed = false;
        let btnBPressStartTime = 0;
        let btnBLongTriggered = false;
        let lastHomeSentTime = 0;

        function triggerGoHome(source) {
            let now = Date.now();
            if (now - lastHomeSentTime < 2000) return;
            lastHomeSentTime = now;
            console.log("[WebXR] Go Home command triggered from VR Controller!");
            if (source && source.gamepad && source.gamepad.hapticActuators && source.gamepad.hapticActuators.length > 0) {
                source.gamepad.hapticActuators[0].pulse(1.0, 300);
                setTimeout(() => {
                    if (source.gamepad.hapticActuators.length > 0) source.gamepad.hapticActuators[0].pulse(0.8, 200);
                }, 350);
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'go_home', home: true, thumbstick_click: true }));
            } else {
                fetch('/api/go_home').catch(() => {});
            }
            let elHome = document.getElementById('val-home-btn');
            if (elHome) {
                elHome.innerText = "Returning Home...";
                elHome.className = "badge green";
                setTimeout(() => {
                    elHome.innerText = "Thumbstick / Hold B (>3s)";
                    elHome.className = "badge gray";
                }, 4000);
            }
        }

        btnEnterVR.addEventListener('click', () => {
            if (btnEnterVR.isVR) {
                if (!xrSession) {
                    // DOM Overlay BẮT BUỘC phải hiển thị (không được display: none) TRƯỚC KHI requestSession
                    vrOverlay.style.display = 'flex';
                    
                    // Yêu cầu chế độ AR với dom-overlay
                    navigator.xr.requestSession('immersive-ar', {
                        optionalFeatures: ['local-floor', 'bounded-floor', 'dom-overlay'],
                        domOverlay: { root: vrOverlay }
                    }).then(onSessionStarted).catch((err) => {
                        console.log("AR failed, fallback to VR", err);
                        navigator.xr.requestSession('immersive-vr', {
                            optionalFeatures: ['local-floor', 'bounded-floor', 'dom-overlay'],
                            domOverlay: { root: vrOverlay }
                        }).then(onSessionStarted);
                    });
                }
            } else {
                // PC Preview mode
                vrOverlay.style.display = 'flex';
                document.getElementById('launcher-screen').style.display = 'none';
                document.body.style.backgroundColor = '#ccc'; // Giả lập nền AR
            }
        });

        function onSessionStarted(session) {
            xrSession = session;
            btnEnterVR.style.display = 'none';
            document.getElementById('launcher-screen').style.display = 'none';
            vrOverlay.style.display = 'flex'; // Hiện Overlay bằng flex để căn giữa
            setDriveMode(true, true); // Luôn mặc định mở Drive Mode ngay khi vào VR để lái robot tức thì
            statusDiv.innerText = "In VR Session";

            session.addEventListener('end', onSessionEnded);

            // Thiết lập WebGL dummy để WebXR chịu chạy loop
            const canvas = document.getElementById('webgl-canvas');
            glContext = canvas.getContext('webgl', { xrCompatible: true, alpha: true });
            session.updateRenderState({ baseLayer: new XRWebGLLayer(session, glContext) });

            session.requestReferenceSpace('local').then((refSpace) => {
                xrRefSpace = refSpace;
                session.requestAnimationFrame(onXRFrame);
            });
        }

        function onSessionEnded(event) {
            xrSession = null;
            btnEnterVR.style.display = 'inline-block';
            vrOverlay.style.display = 'none'; // Ẩn Cockpit UI
            document.getElementById('launcher-screen').style.display = 'flex';
            statusDiv.innerText = "VR Session Ended.";
        }

        function onXRFrame(time, frame) {
            let session = frame.session;
            session.requestAnimationFrame(onXRFrame);
            
            // Xóa buffer WebGL để không cản trở lớp AR (Xuyên thấu)
            if (glContext) {
                glContext.bindFramebuffer(glContext.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
                glContext.clearColor(0, 0, 0, 0); // Trong suốt hoàn toàn
                glContext.clear(glContext.COLOR_BUFFER_BIT | glContext.DEPTH_BUFFER_BIT);
            }

            if (!ws || ws.readyState !== WebSocket.OPEN) return;

            let sentData = false;

            let activeSource = null;
            // Ưu tiên tay phải
            for (let source of session.inputSources) {
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
                    if (source.targetRayMode === 'gaze') continue;
                    if (!source.gripSpace && !source.targetRaySpace) continue;
                    if (source.handedness === 'left') {
                        activeSource = source;
                        break;
                    }
                }
            }

            if (activeSource) {
                activeHandedness = activeSource.handedness;
                let space = activeSource.gripSpace || activeSource.targetRaySpace;
                let pose = frame.getPose(space, xrRefSpace);
                
                if (pose) {
                    let triggerPressed = false;
                    let gripPressed = false;
                    let gripperRawPressed = false;
                    let gripperBtnPressed = false;
                    let driveTogglePressed = false;
                    let thumbstickPressed = false;
                    
                    if (activeSource.gamepad && activeSource.gamepad.buttons) {
                        // Button 0: Index Trigger (bóp trỏ - tịnh tiến)
                        if (activeSource.gamepad.buttons.length > 0) triggerPressed = activeSource.gamepad.buttons[0].pressed;
                        // Button 1: Hand Grip (bóp hông - xoay)
                        if (activeSource.gamepad.buttons.length > 1) gripPressed = activeSource.gamepad.buttons[1].pressed;
                        // Button 3: Thumbstick Click (nhấn cần gạt Analog) -> Lệnh Go Home
                        if (activeSource.gamepad.buttons.length > 3) thumbstickPressed = activeSource.gamepad.buttons[3].pressed;
                        // Button 4: Nút A (tay phải) hoặc Nút X (tay trái) - Kẹp/Nhả Gripper
                        if (activeSource.gamepad.buttons.length > 4) gripperRawPressed = activeSource.gamepad.buttons[4].pressed;
                        // Button 5: Nút B (tay phải) hoặc Nút Y (tay trái) - Phím cứng chuyển đổi Drive Mode hoặc Giữ để Go Home
                        if (activeSource.gamepad.buttons.length > 5) driveTogglePressed = activeSource.gamepad.buttons[5].pressed;
                    }

                    // 1. Xử lý Thumbstick Click (nhấn cần gạt analog R3): Kích hoạt Go Home tức thì
                    if (thumbstickPressed && !lastThumbstickPressed) {
                        triggerGoHome(activeSource);
                    }
                    lastThumbstickPressed = thumbstickPressed;

                    // 2. Xử lý nút B / Y:
                    // - Bấm nhả nhanh (< 3 giây): Bật / Tắt Drive Mode
                    // - Nhấn giữ lâu (> 3 giây): Kích hoạt Go Home
                    if (driveTogglePressed) {
                        if (btnBPressStartTime === 0) {
                            btnBPressStartTime = Date.now();
                            btnBLongTriggered = false;
                        } else if (Date.now() - btnBPressStartTime > 3000 && !btnBLongTriggered) {
                            btnBLongTriggered = true;
                            triggerGoHome(activeSource);
                        }
                    } else {
                        if (btnBPressStartTime > 0) {
                            if (!btnBLongTriggered) {
                                setDriveMode(!uiLocked, true);
                                if (activeSource.gamepad && activeSource.gamepad.hapticActuators && activeSource.gamepad.hapticActuators.length > 0) {
                                    activeSource.gamepad.hapticActuators[0].pulse(0.7, 120);
                                }
                            }
                            btnBPressStartTime = 0;
                            btnBLongTriggered = false;
                        }
                    }

                    // 3. Kiểm tra cả tay cầm phụ nếu có (ví dụ tay trái nhấn cần gạt L3 hoặc nút X)
                    for (let s of session.inputSources) {
                        if (s !== activeSource && s.gamepad && s.gamepad.buttons) {
                            if (s.gamepad.buttons.length > 3 && s.gamepad.buttons[3].pressed) {
                                triggerGoHome(s);
                                thumbstickPressed = true;
                            }
                            if (s.gamepad.buttons.length > 4 && s.gamepad.buttons[4].pressed) {
                                gripperRawPressed = true;
                            }
                        }
                    }

                    // 4. Xử lý cơ chế Toggle Gripper (Bấm 1 lần để kẹp/hút, bấm lần nữa để thả)
                    let gripperToggleTriggered = false;
                    if (gripperRawPressed && !lastGripperRawPressed) {
                        gripperToggleTriggered = true;
                        gripperClosedState = !gripperClosedState;
                        console.log(`[Gripper Toggle] Chuyển trạng thái: ${gripperClosedState ? 'KẸP (CLOSED)' : 'THẢ (OPEN)'}`);
                        if (activeSource.gamepad && activeSource.gamepad.hapticActuators && activeSource.gamepad.hapticActuators.length > 0) {
                            activeSource.gamepad.hapticActuators[0].pulse(gripperClosedState ? 0.9 : 0.4, 120);
                        }
                    }
                    lastGripperRawPressed = gripperRawPressed;
                    gripperBtnPressed = gripperClosedState;

                    let pos = pose.transform.position;
                    let quat = pose.transform.orientation;

                    let headQuat = null;
                    let headPos = null;
                    let viewerPose = frame.getViewerPose(xrRefSpace);
                    if (viewerPose) {
                        let hp = viewerPose.transform.position;
                        let ho = viewerPose.transform.orientation;
                        headPos = [hp.x, hp.y, hp.z];
                        headQuat = [ho.w, ho.x, ho.y, ho.z];
                    }

                    let stickTilted = false;
                    if (activeSource.gamepad && activeSource.gamepad.axes && activeSource.gamepad.axes.length >= 4) {
                        stickTilted = Math.abs(activeSource.gamepad.axes[2]) > 0.15 || Math.abs(activeSource.gamepad.axes[3]) > 0.15;
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
                        gripper_toggle_btn: gripperToggleTriggered,
                        gripper_btn: gripperBtnPressed,
                        button_a: gripperRawPressed,
                        button_b: driveTogglePressed,
                        thumbstick: thumbstickPressed || stickTilted,
                        thumbstick_click: thumbstickPressed,
                        joystick_tilt: stickTilted,
                        home: thumbstickPressed,
                        drive_mode: uiLocked
                    }));
                    
                    let handName = activeSource.handedness === 'right' ? "Right Hand" : (activeSource.handedness === 'left' ? "Left Hand" : "Unknown");
                    
                    // Cập nhật dữ liệu lên URSim Layout
                    elValSource.innerText = handName;
                    elValSource.style.color = "#10b981";
                    elValSource.style.fontWeight = "bold";
                    
                    elValTrigger.innerText = triggerPressed ? "Pressed" : "Released";
                    elValTrigger.className = triggerPressed ? "badge red" : "badge gray";
                    
                    elValGrip.innerText = gripPressed ? "Pressed" : "Released";
                    elValGrip.className = gripPressed ? "badge red" : "badge gray";

                    if (elValGripperBtn) {
                        elValGripperBtn.innerText = gripperBtnPressed ? "Closed (Suction ON)" : "Open (Suction OFF)";
                        elValGripperBtn.className = gripperBtnPressed ? "badge red" : "badge gray";
                    }
                    
                    // elValX, Y, Z, Qx, Qy, Qz, Qw được cập nhật tự động từ luồng telemetry của Robot thực tế (20Hz)
                    sentData = true;
                }
            }

            // Ghi log để debug nếu không có tay cầm nào được nhận diện
            if (!sentData) {
                elValSource.innerText = "Waiting...";
                elValSource.style.color = "#64748b";
                elValSource.style.fontWeight = "normal";
                
                let debugStr = "";
                for(let s of session.inputSources) {
                    debugStr += `[${s.handedness || 'none'}, ${s.targetRayMode}, pose:${frame.getPose(s.gripSpace||s.targetRaySpace, xrRefSpace) ? 'yes':'null'}] `;
                }
                // Optional: Console log the debug string since the UI doesn't have a giant box for it anymore
                // console.log("Missing pose: ", debugStr);
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

        setInterval(pollCameraFrame, 100);

        // Start
        connectWebSocket();

        // Auto-assign the correct IP for Viser iframe based on current host
        const viserIframe = document.getElementById('viser-iframe');
        if (viserIframe) {
            let host = window.location.hostname || '127.0.0.1';
            viserIframe.src = `http://${host}:8080`;
        }
