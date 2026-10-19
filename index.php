<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Распознавание жестов</title>
</head>
<body>
    <h2>Распознавание жестов</h2>
    <video id="video" autoplay muted playsinline style="width:320px; border:1px solid #ccc;"></video>
    <p>Текущий жест: <strong id="gesture">—</strong></p>
    <p>Последние 5: <span id="history">—</span></p>

    <script>
        const video = document.getElementById('video');
        const gestureEl = document.getElementById('gesture');
        const historyEl = document.getElementById('history');

        // 1. Запускаем камеру
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { alert('Не удалось получить камеру: ' + err); });

        // 2. Подключаем WebSocket
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.onopen = () => {
            console.log('WebSocket подключён');
            // Начинаем отправлять кадры
            setInterval(sendFrame, 66); // ~10 FPS, регулируйте при необходимости
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            gestureEl.textContent = data.gesture || '—';
            historyEl.textContent = data.history.join(' ') || '—';
        };

        // 3. Извлекаем кадр с canvas и отправляем
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');

        function sendFrame() {
            if (ws.readyState !== WebSocket.OPEN) return;
            if (video.videoWidth === 0) return;

            // Рисуем текущий кадр видео на canvas (уменьшенный до 224x224)
            ctx.drawImage(video, 0, 0, 224, 224);

            // Конвертируем в JPEG blob и отправляем
            canvas.toBlob(blob => {
                blob.arrayBuffer().then(buf => {
                    ws.send(buf);
                });
            }, 'image/jpeg', 0.7);
        }
    </script>
</body>
</html>