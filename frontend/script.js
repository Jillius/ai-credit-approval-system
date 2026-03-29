document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('credit-form');
    const submitBtn = document.getElementById('submit-btn');
    const statusDiv = document.getElementById('ai-status');
    const overlay = document.getElementById('result-overlay');
    const closeBtn = document.getElementById('close-result');
    const resetBtn = document.getElementById('reset-btn');
    const titleStatus = document.getElementById('result-title');
    const resultCard = document.querySelector('.result-card');
    const fillBar = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const resultDesc = document.getElementById('result-desc');

    let modelReady = false;
    let expectedFeatures = [];

    // Check backend status
    const pollStatus = setInterval(async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/status');
            const data = await response.json();
            
            if (data.ready) {
                modelReady = true;
                expectedFeatures = data.features;
                statusDiv.className = 'ai-status ready';
                statusDiv.innerHTML = `✅ AI Core Online (Test Acc: ${(data.accuracy * 100).toFixed(1)}%)`;
                submitBtn.disabled = false;
                clearInterval(pollStatus);
            } else if (data.error) {
                statusDiv.className = 'ai-status error';
                statusDiv.innerHTML = `❌ AI Error: ${data.error.substring(0, 50)}... Please check terminal.`;
                submitBtn.disabled = true;
                clearInterval(pollStatus);
                alert("The AI model failed to initialize! Error: " + data.error);
            } else {
                statusDiv.innerHTML = '⚙️ AI Model Initializing...';
                submitBtn.disabled = true;
            }
        } catch (error) {
            statusDiv.className = 'ai-status error';
            statusDiv.innerHTML = 'Backend currently offline. Run start.bat';
            submitBtn.disabled = true;
        }
    }, 2000);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!modelReady) return;

        // Button Loading State
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const dataPayload = {};
        
        formData.forEach((value, key) => {
            // Convert to numbers if they are strictly numerical strings
            dataPayload[key] = !isNaN(value) && value !== '' ? Number(value) : value;
        });

        // Ensure all 20 features are sent (use defaults if missing)
        // We defined all 20 in the HTML including hidden ones.
        
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataPayload)
            });

            const result = await response.json();

            if (result.status === 'success') {
                showResult(result.prediction, result.confidence);
            } else {
                alert(`Error: ${result.error}`);
                resetBtnState();
            }

        } catch (error) {
            alert('Failed to connect to backend AI server. Check console for details.');
            resetBtnState();
        }
    });

    function showResult(prediction, confidence) {
        // Remove old classes
        resultCard.classList.remove('approved', 'rejected');
        fillBar.style.width = '0%';

        const isApproved = prediction === 'Approved';
        
        // Populate modal
        resultCard.classList.add(isApproved ? 'approved' : 'rejected');
        titleStatus.innerText = isApproved ? 'Credit Approved' : 'Credit Rejected';
        resultDesc.innerText = isApproved 
            ? 'The TabPFN AI model considers this application low-risk based on historical German Credit data attributes.'
            : 'The TabPFN AI model has flagged this application as high-risk. Approval is not recommended.';
        
        const perc = (confidence * 100).toFixed(1);
        confidenceText.innerText = `AI Certainty: ${perc}%`;

        // Show overlay
        overlay.classList.remove('hidden');

        // Animate meter bar after a small delay for transition
        setTimeout(() => {
            fillBar.style.width = `${perc}%`;
        }, 300);

        resetBtnState();
    }

    function resetBtnState() {
        submitBtn.classList.remove('loading');
        submitBtn.disabled = false;
    }

    closeBtn.addEventListener('click', () => {
        overlay.classList.add('hidden');
    });

    resetBtn.addEventListener('click', () => {
        overlay.classList.add('hidden');
        form.reset();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Initial check
    statusDiv.innerHTML = '⚙️ AI Model Initializing...';
    submitBtn.disabled = true;
});
