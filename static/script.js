 let model = null, audioContext = null, mediaRecorder = null, audioChunks = [], currentAudioBuffer = null, isRecording = false;

        // DOM
        const fileInput = document.getElementById('fileInput'),
            uploadBtn = document.getElementById('uploadBtn'),
            recordBtn = document.getElementById('recordBtn'),
            stopBtn = document.getElementById('stopBtn'),
            analyzeBtn = document.getElementById('analyzeBtn'),
            waveform = document.getElementById('waveform'),
            waveCanvas = document.getElementById('waveCanvas'),
            loader = document.getElementById('loader'),
            status = document.getElementById('status'),
            result = document.getElementById('result'),
            resultBadge = document.getElementById('resultBadge'),
            confidence = document.getElementById('confidence'),
            progressFill = document.getElementById('progressFill'),
            audioPlayer = document.getElementById('audioPlayer');

       const CONFIG = { SAMPLE_RATE: 16000, DURATION: 3 };


        
        // AUDIO CONTEXT
        function initAudioContext() {
            if (!audioContext) { audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CONFIG.SAMPLE_RATE }); }
        }

        // CONVERT AUDIOBUFFER TO WAV
        function audioBufferToWav(buffer) {
            const numOfChan = buffer.numberOfChannels,
                length = buffer.length * numOfChan * 2 + 44,
                bufferArray = new ArrayBuffer(length),
                view = new DataView(bufferArray),
                channels = [],
                sampleRate = buffer.sampleRate;

            let offset = 0;

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            writeString(view, 0, 'RIFF'); view.setUint32(4, length - 8, true); writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
            view.setUint16(22, numOfChan, true); view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numOfChan * 2, true); view.setUint16(32, numOfChan * 2, true);
            view.setUint16(34, 16, true); writeString(view, 36, 'data'); view.setUint32(40, length - 44, true);

            for (let i = 0; i < numOfChan; i++) { channels.push(buffer.getChannelData(i)); }

            let pos = 44;
            for (let i = 0; i < buffer.length; i++) {
                for (let ch = 0; ch < numOfChan; ch++) {
                    let sample = Math.max(-1, Math.min(1, channels[ch][i]));
                    view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
                    pos += 2;
                }
            }
            return bufferArray;
        }

        // LOAD AUDIO INTO PLAYER
        async function loadAudioForPlayback(audioBuffer) {
            const wavArrayBuffer = await audioBufferToWav(audioBuffer);
            const blob = new Blob([wavArrayBuffer], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            audioPlayer.src = url;
            audioPlayer.style.display = 'block';
        }

        // FILE UPLOAD
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async e => {
            const file = e.target.files[0]; if (!file) return;
            resetResult(); status.innerHTML = '📂 Loading audio file...'; loader.classList.add('active');
            try {
                const arrayBuffer = await file.arrayBuffer();
                initAudioContext();
                currentAudioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                drawWaveform(currentAudioBuffer);
                loadAudioForPlayback(currentAudioBuffer);
                analyzeBtn.disabled = false;
                status.innerHTML = '✅ Audio loaded! Click "Analyze Now"';
            } catch (err) { status.innerHTML = '❌ Error loading audio file.'; console.error(err); }
            finally { loader.classList.remove('active'); }
        });

        // RECORDING
        recordBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);

        async function startRecording() {
            if (isRecording) return;
            resetResult();
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream); audioChunks = [];
                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    const blob = new Blob(audioChunks, { type: 'audio/wav' });
                    const arrayBuffer = await blob.arrayBuffer();
                    initAudioContext();
                    currentAudioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    drawWaveform(currentAudioBuffer);
                    loadAudioForPlayback(currentAudioBuffer);
                    analyzeBtn.disabled = false;
                    status.innerHTML = '✅ Recording complete! Click "Analyze Now".';
                };
                mediaRecorder.start(); isRecording = true;
                recordBtn.disabled = true; stopBtn.disabled = false; recordBtn.classList.add('recording');
                status.innerHTML = '🎙️ Recording... Speak now!';
            } catch (err) { status.innerHTML = '❌ Error accessing microphone.'; console.error(err); }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                recordBtn.disabled = false; stopBtn.disabled = true; recordBtn.classList.remove('recording');
            }
        }

        // WAVEFORM DRAW
        function drawWaveform(audioBuffer) {
            waveform.classList.add('active');
            const canvas = waveCanvas; const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2; canvas.height = canvas.offsetHeight * 2;
            const data = audioBuffer.getChannelData(0);
            const step = Math.ceil(data.length / canvas.width); const amp = canvas.height / 2;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, '#06ffa5'); gradient.addColorStop(0.5, '#3a86ff'); gradient.addColorStop(1, '#8338ec');
            ctx.strokeStyle = gradient; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.beginPath();
            for (let i = 0; i < canvas.width; i++) {
                const min = Math.min(...data.slice(i * step, (i + 1) * step));
                const max = Math.max(...data.slice(i * step, (i + 1) * step));
                ctx.moveTo(i, (1 + min) * amp); ctx.lineTo(i, (1 + max) * amp);
            } ctx.stroke();
        }

        // FEATURE EXTRACTION
        function extractFeatures(audioBuffer) {
            const channelData = audioBuffer.getChannelData(0);
            const targetLength = CONFIG.SAMPLE_RATE * CONFIG.DURATION;
            let processedData = new Float32Array(targetLength);
            if (channelData.length < targetLength) { processedData.set(channelData); } else { processedData = channelData.slice(0, targetLength); }
            const features = []; const frameSize = 2048; const hopSize = CONFIG.HOP_LENGTH;
            for (let i = 0; i < processedData.length - frameSize; i += hopSize) {
                const frame = processedData.slice(i, i + frameSize);
                const energy = frame.reduce((sum, val) => sum + val * val, 0) / frameSize;
                features.push(energy);
            }
            return features;
        }

        // ANALYZE AUDIO
        analyzeBtn.addEventListener('click', async () => {
    if (!currentAudioBuffer) return;

    status.innerHTML = '🔍 Analyzing audio...';
    loader.classList.add('active');
    analyzeBtn.disabled = true;
    result.classList.remove('active');

   try {
    // AudioBuffer → WAV
    const wavBuffer = await audioBufferToWav(currentAudioBuffer);
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });

    // FormData
    const formData = new FormData();
    formData.append("file", blob, "audio.wav");

    // CALL BACKEND
    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    });

    // Parse JSON from the backend
    const data = await response.json();

    if (data.label) {
        // Prediction succeeded
        let label = data.label.toLowerCase();
        let confidence = Number(data.confidence.toFixed(2));
        displayResult({ class: label, confidence: confidence });
    } else if (data.error) {
        // Backend returned an error
        console.error("Prediction failed:", data.error);
        alert("Prediction failed: " + data.error);
    } else {
        // Unexpected response
        console.error("Unexpected response:", data);
        alert("Prediction failed: Unknown error");
    }

} catch (err) {
    console.error(err);
    status.innerHTML = '❌ Backend error';
} finally {
    loader.classList.remove('active');
    analyzeBtn.disabled = false;
}

});


        // DISPLAY RESULT
        function displayResult(prediction) {
            resultBadge.textContent = prediction.class.toUpperCase();
            resultBadge.className = `result-badge ${prediction.class}`;
            const emoji = prediction.class === 'human' ? '👤' : '🤖';
            confidence.innerHTML = `${emoji} Confidence: ${prediction.confidence.toFixed(1)}%`;
            progressFill.style.width = `${prediction.confidence}%`;
            result.classList.add('active');
            const resultEmoji = prediction.class === 'human' ? '✅' : '⚠️';
            status.innerHTML = `${resultEmoji} Classification complete: ${prediction.class.toUpperCase()} VOICE DETECTED`;
        }

        // RESET RESULT
        function resetResult() {
            result.classList.remove('active');
            progressFill.style.width = '0%';
            confidence.innerHTML = '';
        }

        // INIT
       // window.addEventListener('load', () => { loadModel(); });