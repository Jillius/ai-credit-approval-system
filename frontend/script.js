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
    let currentTheme = 'default';
    let currencyRates = null;
    let selectedCurrency = 'USD';

    // Elements
    const modelSelector = document.getElementById('model_id');
    const model1Fields = document.getElementById('model-1-fields');
    const model2Fields = document.getElementById('model-2-fields');
    
    // Settings Elements
    const settingsBtn = document.getElementById('settings-btn');
    const settingsSidebar = document.getElementById('settings-sidebar');
    const closeSettings = document.getElementById('close-settings');
    const themeSelector = document.getElementById('theme-selector');
    const currencySelector = document.getElementById('currency-selector');
    
    // Inputs that need currency conversion
    const creditAmountInput = document.getElementById('credit_amount');
    const loanAmntInput = document.getElementById('loan_amnt');
    const personIncomeInput = document.getElementById('person_income');
    
    const creditHint = document.getElementById('credit_amount_hint');
    const loanHint = document.getElementById('loan_amnt_hint');
    const incomeHint = document.getElementById('person_income_hint');

    // Sidebar Toggle
    settingsBtn.addEventListener('click', () => settingsSidebar.classList.add('open'));
    closeSettings.addEventListener('click', () => settingsSidebar.classList.remove('open'));

    // Theme Toggle
    themeSelector.addEventListener('change', (e) => {
        currentTheme = e.target.value;
        document.body.classList.remove('theme-laotse', 'theme-bank');
        document.getElementById('bank-logo').classList.add('hidden');
        if (currentTheme === 'laotse') document.body.classList.add('theme-laotse');
        if (currentTheme === 'bank') {
            document.body.classList.add('theme-bank');
            document.getElementById('bank-logo').classList.remove('hidden');
        }
    });

    // Model Selector
    modelSelector.addEventListener('change', (e) => {
        currentModelId = e.target.value;
        if (currentModelId === 'model_2') {
            model1Fields.style.display = 'none';
            model2Fields.style.display = 'block';
        } else {
            model1Fields.style.display = 'block';
            model2Fields.style.display = 'none';
        }
        updateCurrencyHints();
    });

    // Currency Initialization & Live API
    async function initCurrencyAPI() {
        try {
            const res = await fetch('https://open.er-api.com/v6/latest/EUR');
            const data = await res.json();
            currencyRates = data.rates;
            updateCurrencyHints();
        } catch (e) {
            console.error('Failed to fetch currency rates', e);
        }
    }
    initCurrencyAPI();

    currencySelector.addEventListener('change', (e) => {
        selectedCurrency = e.target.value;
        updateCurrencyHints();
    });

    [creditAmountInput, loanAmntInput, personIncomeInput].forEach(inp => {
        inp.addEventListener('input', updateCurrencyHints);
    });

    function updateCurrencyHints() {
        if (!currencyRates) return;
        
        const rate = currencyRates[selectedCurrency];
        
        if (currentModelId === 'model_1') {
            const amount = parseFloat(creditAmountInput.value) || 0;
            const eurAmount = amount / rate;
            const dmAmount = Math.round(eurAmount / 0.86); // 1 DM (1994) = 0.86 EUR (2026 inflation)
            creditHint.innerText = `≈ ${dmAmount.toLocaleString()} DM (1994 Purchasing Power)`;
        } else {
            // Model 2
            const usdRate = currencyRates['USD'];
            const loanAmnt = parseFloat(loanAmntInput.value) || 0;
            const incomeAmnt = parseFloat(personIncomeInput.value) || 0;
            
            const loanUsd = Math.round((loanAmnt / rate) * usdRate);
            const incomeUsd = Math.round((incomeAmnt / rate) * usdRate);
            
            loanHint.innerText = `≈ $${loanUsd.toLocaleString()} USD (Model Native)`;
            incomeHint.innerText = `≈ $${incomeUsd.toLocaleString()} USD (Model Native)`;
        }
    }

    const pollStatus = setInterval(async () => {
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', { method: 'OPTIONS' });
            statusDiv.className = 'ai-status ready';
            statusDiv.innerHTML = ` Hybrid AI System Online `;
            submitBtn.disabled = false;
            modelReady = true;
        } catch (error) {
            statusDiv.className = 'ai-status connecting';
            statusDiv.innerHTML = ' Waiting for AI Models...';
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
                let value = input.value;
                if (!isNaN(value) && value.trim() !== '') {
                    value = Number(value);
                    
                    // Apply Algorithm Conversions before sending
                    if (currencyRates) {
                        const rate = currencyRates[selectedCurrency];
                        const usdRate = currencyRates['USD'];
                        if (currentModelId === 'model_1' && input.name === 'credit_amount') {
                            value = Math.round((value / rate) / 0.86); // To DM
                        }
                        if (currentModelId === 'model_2' && (input.name === 'loan_amnt' || input.name === 'person_income')) {
                            value = Math.round((value / rate) * usdRate); // To USD
                        }
                    }
                }
                dataPayload[input.name] = value;
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
    fillBar.style.width = '0%';

    const score = result.score * 100;
    const isApproved = result.final_decision === 'Approved';
    resultCard.classList.add(isApproved ? 'approved' : 'rejected');
    
    
    titleStatus.innerText = isApproved ? 'Credit Approved' : 'Credit Rejected';

    
    let message = "";
    if (score < 40) {
        message = "High Risk: Your financial profile does not meet the requirements.";
    } else if (score >= 40 && score < 50) {
        message = "Borderline Risk: Credit rejected, but human review could change the outcome.";
    } else if (score >= 50 && score <= 65) {
        message = "Conditional Approval: Borderline case, extra documents may be requested.";
    } else {
        message = "Strong Approval: Your financial profile indicates high reliability.";
    }

    
    resultDesc.innerHTML = `${message}<br><small style="opacity: 0.7; display: block; margin-top: 10px;">
        AI Analysis: TabPFN (${result.tabpfn_result}) | CatBoost (${result.catboost_result})</small>`;
    
    
    const perc = score.toFixed(1);
    confidenceText.innerText = `Final Approval Probability: ${perc}%`;

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