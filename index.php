<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Распознавание жестов</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        #video { width: 320px; height: 240px; background: #000; border-radius: .5rem; }
        #history { word-break: break-word; }
        #toast { position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%); z-index: 1080; }
    </style>
</head>
<body class="bg-light">

    <div class="container py-4" style="max-width: 640px;">

        <h2 class="text-center mb-4">Распознавание жестов</h2>

        <!-- Видео по центру -->
        <div class="d-flex justify-content-center mb-3">
            <div class="card shadow-sm">
                <div class="card-body p-2">
                    <video id="video" autoplay muted playsinline></video>
                </div>
            </div>
        </div>

        <!-- Кнопки по центру, все в одном тоне -->
        <div class="d-flex justify-content-center gap-2 flex-wrap mb-3">
            <button id="startBtn" class="btn btn-primary">Старт</button>
            <button id="copyBtn"  class="btn btn-primary" disabled>Копировать текст</button>
            <button id="resetBtn" class="btn btn-primary" disabled>Сбросить историю</button>
        </div>

        <!-- Статус -->
        <div class="text-center mb-3">
            <span id="status" class="badge text-bg-secondary">Статус: ожидание запуска</span>
        </div>

        <!-- Результаты -->
        <div class="card shadow-sm">
            <div class="card-body text-center">
                <p class="mb-2">Текущий жест: <strong id="gesture" class="fs-5">—</strong></p>
                <p class="mb-0">Последние 5: <span id="history" class="fw-semibold">—</span></p>
            </div>
        </div>

    </div>

    <!-- Тосты -->
    <div id="toast" class="toast align-items-center text-bg-dark border-0" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="d-flex">
            <div class="toast-body" id="toastBody"></div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const video     = document.getElementById('video');
        const gestureEl = document.getElementById('gesture');
        const historyEl = document.getElementById('history');
        const statusEl  = document.getElementById('status');
        const startBtn  = document.getElementById('startBtn');
        const copyBtn   = document.getElementById('copyBtn');
        const resetBtn  = document.getElementById('resetBtn');
        const toastEl   = document.getElementById('toast');
        const toastBody = document.getElementById('toastBody');
        const toast     = new bootstrap.Toast(toastEl, { delay: 1500 });

        let ws = null;
        let stream = null;
        let sendTimer = null;
        let started = false;

        let lastText = '';
        let localHistory = [];
        let lastSeenGesture = '';

        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');

        function setStatus(text) {
            statusEl.textContent = 'Статус: ' + text;
        }

        function showToast(msg) {
            toastBody.textContent = msg;
            toast.show();
        }

        function renderHistory() {
            historyEl.textContent = localHistory.length ? localHistory.join(' ') : '—';
        }

        async function start() {
            if (started) return;
            started = true;
            startBtn.disabled = true;
            setStatus('запрос камеры…');

            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                await video.play().catch(() => {});
            } catch (err) {
                setStatus('камера недоступна: ' + err.message);
                started = false;
                startBtn.disabled = false;
                alert('Не удалось получить камеру: ' + err);
                return;
            }

            setStatus('подключение к серверу…');
            connectWebSocket();
        }

        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');

            ws.onopen = () => {
                console.log('WebSocket подключён');
                setStatus('работает');
                copyBtn.disabled = false;
                resetBtn.disabled = false;
                startSendLoop();
            };

            ws.onmessage = (event) => {
                let data;
                try { data = JSON.parse(event.data); }
                catch { return; }

                const text = data.text ?? data.gesture ?? '';
                if (text) lastText = text;
                gestureEl.textContent = lastText || '—';

                // Историю собираем ТОЛЬКО локально из новых gesture.
                // "no" (нет жеста) в историю не добавляем.
                const g = data.gesture;
                if (g && g !== lastSeenGesture && g.toLowerCase() !== 'no') {
                    lastSeenGesture = g;
                    localHistory.push(g);
                    localHistory = localHistory.slice(-5);
                    renderHistory();
                }
            };

            ws.onerror = () => setStatus('ошибка WebSocket');
            ws.onclose = () => {
                setStatus('соединение закрыто');
                stopSendLoop();
                copyBtn.disabled = true;
                resetBtn.disabled = true;
            };
        }

        function startSendLoop() {
            stopSendLoop();
            sendTimer = setInterval(sendFrame, 100);
        }

        function stopSendLoop() {
            if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
        }

        function sendFrame() {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            if (video.videoWidth === 0) return;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                if (!blob) return;
                blob.arrayBuffer().then(buf => {
                    if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
                });
            }, 'image/jpeg', 0.7);
        }

        async function copyText() {
            const text = localHistory.join(' ').trim();
            if (!text) { showToast('История пуста'); return; }
            try {
                await navigator.clipboard.writeText(text);
                showToast('Скопировано: ' + text);
            } catch {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                ta.remove();
                showToast('Скопировано: ' + text);
            }
        }

        function resetHistory() {
            localHistory = [];
            lastText = '';
            lastSeenGesture = '';
            gestureEl.textContent = '—';
            renderHistory();
            showToast('История сброшена');
        }

        startBtn.addEventListener('click', start);
        copyBtn.addEventListener('click', copyText);
        resetBtn.addEventListener('click', resetHistory);

        renderHistory();
    </script>
</body>
</html>