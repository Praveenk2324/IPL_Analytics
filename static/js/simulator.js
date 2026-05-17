/**
 * IPL Analytics — Live Match Simulator Page
 */

const Simulator = {
    initialized: false,

    init() {
        if (!this.initialized) this.render();
        this.initialized = true;
    },

    render() {
        const page = document.getElementById('page-simulator');
        page.innerHTML = `
            <h1 class="page-title">🎯 Live Match Simulator</h1>
            <p class="page-subtitle">Input the current match state to get real-time win probability from our ML models</p>
            <div class="grid-sidebar">
                <div class="card" id="sim-controls">
                    <h3 class="section-title">⚙️ Match State</h3>
                    <div class="form-group">
                        <label class="form-label">Batting Team</label>
                        <select class="form-select" id="sim-batting-team"></select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Bowling Team</label>
                        <select class="form-select" id="sim-bowling-team"></select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Target Score</label>
                        <div class="range-wrapper">
                            <input type="range" class="form-range" id="sim-target" min="80" max="300" value="180">
                            <span class="range-value" id="sim-target-val">180</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Current Score</label>
                        <div class="range-wrapper">
                            <input type="range" class="form-range" id="sim-score" min="0" max="299" value="85">
                            <span class="range-value" id="sim-score-val">85</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Overs Completed</label>
                        <div class="range-wrapper">
                            <input type="range" class="form-range" id="sim-overs" min="1" max="20" value="10" step="1">
                            <span class="range-value" id="sim-overs-val">10</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Wickets Fallen</label>
                        <div class="range-wrapper">
                            <input type="range" class="form-range" id="sim-wickets" min="0" max="9" value="3">
                            <span class="range-value" id="sim-wickets-val">3</span>
                        </div>
                    </div>
                    <button class="btn btn-primary btn-full" id="sim-predict-btn">
                        🔮 Predict Win Probability
                    </button>
                </div>
                <div>
                    <div class="card" id="sim-results">
                        <div class="empty-state">
                            <div class="empty-state-icon">🏏</div>
                            <p>Set the match state and click predict to see win probabilities</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        this.setupControls();
    },

    setupControls() {
        const batSelect = document.getElementById('sim-batting-team');
        const bowlSelect = document.getElementById('sim-bowling-team');
        App.populateSelect(batSelect, App.teams, 'Select batting team');
        App.populateSelect(bowlSelect, App.teams, 'Select bowling team');

        if (App.teams.length >= 2) {
            batSelect.value = App.teams[0];
            bowlSelect.value = App.teams[1];
        }

        // Range slider live updates
        ['target', 'score', 'overs', 'wickets'].forEach(id => {
            const el = document.getElementById(`sim-${id}`);
            const val = document.getElementById(`sim-${id}-val`);
            el.addEventListener('input', () => val.textContent = el.value);
        });

        document.getElementById('sim-predict-btn').addEventListener('click', () => this.predict());
    },

    async predict() {
        const btn = document.getElementById('sim-predict-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner" style="padding:0"></span> Predicting...';

        const payload = {
            target: +document.getElementById('sim-target').value,
            current_score: +document.getElementById('sim-score').value,
            overs_completed: +document.getElementById('sim-overs').value,
            wickets_fallen: +document.getElementById('sim-wickets').value,
        };

        try {
            const res = await fetch('/api/predict-win', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            this.renderResults(data, payload);
        } catch (e) {
            App.toast('Prediction failed: ' + e.message, 'error');
        }

        btn.disabled = false;
        btn.innerHTML = '🔮 Predict Win Probability';
    },

    renderResults(data, payload) {
        const resultsDiv = document.getElementById('sim-results');
        const battingTeam = document.getElementById('sim-batting-team').value || 'Chasing Team';
        const bowlingTeam = document.getElementById('sim-bowling-team').value || 'Defending Team';

        // Pick XGBoost as primary, fallback to RF
        const primary = data.xgboost || data.random_forest || { chasing_win_prob: 50, defending_win_prob: 50 };
        const chasingProb = primary.chasing_win_prob;
        const ms = data.match_state || {};

        resultsDiv.innerHTML = `
            <h3 class="section-title">📊 Win Probability</h3>
            <div class="gauge-container">
                <div style="text-align:center">
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.5rem">${battingTeam} (Chasing)</div>
                    ${this.renderGauge(chasingProb)}
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.75rem">${bowlingTeam} (Defending): ${(100 - chasingProb).toFixed(1)}%</div>
                </div>
            </div>
            <div class="model-cards">
                ${data.random_forest ? `
                    <div class="model-card">
                        <div class="model-card-name">Random Forest</div>
                        <div class="model-card-value ${data.random_forest.chasing_win_prob > 50 ? 'win' : 'lose'}">${data.random_forest.chasing_win_prob}%</div>
                        <div class="model-card-sub">Chasing team</div>
                    </div>
                ` : ''}
                ${data.xgboost ? `
                    <div class="model-card">
                        <div class="model-card-name">XGBoost</div>
                        <div class="model-card-value ${data.xgboost.chasing_win_prob > 50 ? 'win' : 'lose'}">${data.xgboost.chasing_win_prob}%</div>
                        <div class="model-card-sub">Chasing team</div>
                    </div>
                ` : ''}
            </div>
            <div class="match-state">
                <div class="state-item">
                    <div class="state-value">${ms.current_run_rate || '-'}</div>
                    <div class="state-label">Current RR</div>
                </div>
                <div class="state-item">
                    <div class="state-value">${ms.required_run_rate || '-'}</div>
                    <div class="state-label">Required RR</div>
                </div>
                <div class="state-item">
                    <div class="state-value">${ms.runs_remaining || '-'}</div>
                    <div class="state-label">Runs Left</div>
                </div>
                <div class="state-item">
                    <div class="state-value">${ms.balls_remaining || '-'}</div>
                    <div class="state-label">Balls Left</div>
                </div>
                <div class="state-item">
                    <div class="state-value">${ms.match_phase || '-'}</div>
                    <div class="state-label">Phase</div>
                </div>
            </div>
        `;
    },

    renderGauge(value) {
        const angle = (value / 100) * 180;
        const rad = (angle * Math.PI) / 180;
        const x = 110 + 90 * Math.cos(Math.PI - rad);
        const y = 110 - 90 * Math.sin(Math.PI - rad);
        const largeArc = angle > 90 ? 1 : 0;
        const color = value > 60 ? '#4caf50' : value > 40 ? '#ffa726' : '#ef5350';
        return `
            <svg viewBox="0 0 220 130" width="260" height="150">
                <path d="M 20 110 A 90 90 0 0 1 200 110" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="14" stroke-linecap="round"/>
                <path d="M 20 110 A 90 90 0 ${largeArc} 1 ${x.toFixed(1)} ${y.toFixed(1)}" fill="none" stroke="url(#gaugeGrad)" stroke-width="14" stroke-linecap="round">
                    <animate attributeName="d" from="M 20 110 A 90 90 0 0 1 20 110" to="M 20 110 A 90 90 0 ${largeArc} 1 ${x.toFixed(1)} ${y.toFixed(1)}" dur="0.8s" fill="freeze"/>
                </path>
                <defs><linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#667eea"/><stop offset="100%" stop-color="${color}"/></linearGradient></defs>
                <text x="110" y="100" text-anchor="middle" fill="${color}" font-size="32" font-weight="900" font-family="Inter">${value.toFixed(1)}%</text>
                <text x="110" y="120" text-anchor="middle" fill="#5c6bc0" font-size="10" font-weight="600" font-family="Inter">WIN PROBABILITY</text>
            </svg>
        `;
    },
};
