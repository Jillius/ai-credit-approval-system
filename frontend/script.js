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
    let currentModelId = 'model_1';

    const modelSelector = document.getElementById('model_id');
    const model1Fields = document.getElementById('model-1-fields');
    const model2Fields = document.getElementById('model-2-fields');

    modelSelector.addEventListener('change', (e) => {
        currentModelId = e.target.value;
        if (currentModelId === 'model_2') {
            document.body.classList.add('theme-laotse');
            model1Fields.style.display = 'none';
            model2Fields.style.display = 'block';
        } else {
            document.body.classList.remove('theme-laotse');
            model1Fields.style.display = 'block';
            model2Fields.style.display = 'none';
        }
    });

    const pollStatus = setInterval(async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', { method: 'OPTIONS' });
            statusDiv.className = 'ai-status ready';
            statusDiv.innerHTML = `✅ Hybrid AI System Online (TabPFN + CatBoost)`;
            submitBtn.disabled = false;
            modelReady = true;
        } catch (error) {
            statusDiv.className = 'ai-status connecting';
            statusDiv.innerHTML = '⚙️ Waiting for AI Models...';
            submitBtn.disabled = true;
            modelReady = false;
        }
    }, 3000);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!modelReady) return;

        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        const dataPayload = {};
        const activeFieldsContainer = (currentModelId === 'model_1') ? model1Fields : model2Fields;
        const inputs = activeFieldsContainer.querySelectorAll('input, select, textarea');
        
        inputs.forEach(input => {
            if (input.name && input.name !== 'loan_grade') {
                const value = input.value;
                dataPayload[input.name] = !isNaN(value) && value.trim() !== '' ? Number(value) : value;
            }
        });
        
        const finalPayload = {
            model_id: currentModelId,
            data: dataPayload
        };
        
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(finalPayload)
            });

            const result = await response.json();

            if (result.status === 'success') {
                showResult(result);
            } else {
                alert(`Error: ${result.error}`);
                resetBtnState();
            }
        } catch (error) {
            alert('Failed to connect to backend AI server.');
            resetBtnState();
        }
    });

    function showResult(result) {
        resultCard.classList.remove('approved', 'rejected');
        resultDesc.classList.remove('conflict-warning');
        fillBar.style.width = '0%';

        const isApproved = result.final_decision === 'Approved';
        
        resultCard.classList.add(isApproved ? 'approved' : 'rejected');
        titleStatus.innerText = isApproved ? 'Credit Approved' : 'Credit Rejected';

        let detailText = `Analysis: TabPFN (${result.tabpfn_result}) and CatBoost (${result.catboost_result}) evaluated this application.`;
        if (result.conflict) {
            detailText += " Notice: High model disagreement. Manual verification advised.";
            resultDesc.classList.add('conflict-warning');
        }
        resultDesc.innerText = detailText;
        
        const perc = (result.score * 100).toFixed(1);
        confidenceText.innerText = `System Approval Score: ${perc}%`;

        overlay.classList.remove('hidden');

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
});