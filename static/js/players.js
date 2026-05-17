/**
 * IPL Analytics — Player Performance Dashboard Page
 */

const Players = {
    initialized: false,
    activeTab: 'batting',
    bowlerLoaded: false,

    init() {
        if (!this.initialized) this.render();
        this.initialized = true;
    },

    render() {
        const page = document.getElementById('page-players');
        page.innerHTML = `
            <h1 class="page-title">🏅 Player Performance Dashboard</h1>
            <p class="page-subtitle">Season-wise batting averages, strike rates, bowling economy, and player clustering</p>
            <div class="tabs">
                <button class="tab-btn active" data-tab="batting" id="tab-batting">🏏 Batting</button>
                <button class="tab-btn" data-tab="bowling" id="tab-bowling">🎳 Bowling</button>
                <button class="tab-btn" data-tab="clusters" id="tab-clusters">📊 Player Clusters</button>
            </div>
            <div class="tab-content active" id="content-batting">
                <div class="form-group" style="max-width:400px;margin-bottom:1.5rem">
                    <label class="form-label">Select Batsman</label>
                    <select class="form-select" id="player-bat-select"></select>
                </div>
                <div id="bat-stats-area"><div class="spinner"></div></div>
            </div>
            <div class="tab-content" id="content-bowling">
                <div class="form-group" style="max-width:400px;margin-bottom:1.5rem">
                    <label class="form-label">Select Bowler</label>
                    <select class="form-select" id="player-bowl-select"></select>
                </div>
                <div id="bowl-stats-area"><div class="spinner"></div></div>
            </div>
            <div class="tab-content" id="content-clusters">
                <div id="cluster-area"><div class="spinner"></div></div>
            </div>
        `;
        this.setupTabs();
        this.setupSelects();
    },

    setupTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(`content-${btn.dataset.tab}`).classList.add('active');
                if (btn.dataset.tab === 'clusters') this.loadClusters();
                if (btn.dataset.tab === 'bowling' && !this.bowlerLoaded) {
                    const bowlSelect = document.getElementById('player-bowl-select');
                    if (bowlSelect.value) this.loadBowler(bowlSelect.value);
                }
            });
        });
    },

    setupSelects() {
        const batSelect = document.getElementById('player-bat-select');
        const bowlSelect = document.getElementById('player-bowl-select');
        App.populateSelect(batSelect, App.players.batsmen, 'Select batsman');
        App.populateSelect(bowlSelect, App.players.bowlers, 'Select bowler');

        // Default selections
        const defaultBat = App.players.batsmen.find(p => p.includes('Kohli')) || App.players.batsmen[0];
        const defaultBowl = App.players.bowlers.find(p => p.includes('Bumrah')) || App.players.bowlers[0];
        if (defaultBat) { batSelect.value = defaultBat; this.loadBatsman(defaultBat); }
        if (defaultBowl) { bowlSelect.value = defaultBowl; }

        batSelect.addEventListener('change', () => { if (batSelect.value) this.loadBatsman(batSelect.value); });
        bowlSelect.addEventListener('change', () => { if (bowlSelect.value) this.loadBowler(bowlSelect.value); });
    },

    async loadBatsman(name) {
        const area = document.getElementById('bat-stats-area');
        area.innerHTML = '<div class="spinner"></div>';
        try {
            const res = await fetch(`/api/player-stats/batsman/${encodeURIComponent(name)}`).then(r => r.json());
            if (res.error) throw new Error(res.error);
            const s = res.stats;
            const topRes = await fetch('/api/player-stats/top-batsmen?metric=strike_rate&min_balls=500&n=10').then(r => r.json());

            area.innerHTML = `
                <div class="kpi-grid">
                    <div class="kpi-card"><div class="kpi-value">${Math.round(s.runs).toLocaleString()}</div><div class="kpi-label">Total Runs</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.strike_rate}</div><div class="kpi-label">Strike Rate</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.average}</div><div class="kpi-label">Average</div></div>
                    <div class="kpi-card"><div class="kpi-value">${Math.round(s.balls_faced).toLocaleString()}</div><div class="kpi-label">Balls Faced</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.fours || 0}</div><div class="kpi-label">Fours</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.sixes || 0}</div><div class="kpi-label">Sixes</div></div>
                </div>
                <div class="grid-2">
                    <div class="card"><h3 class="section-title">📈 Runs Per Season</h3><div class="chart-container" id="bat-season-chart"></div></div>
                    <div class="card"><h3 class="section-title">🏆 Top 10 by Strike Rate</h3><div class="chart-container" id="bat-top-chart"></div></div>
                </div>
            `;

            // Season chart
            if (res.season_runs && res.season_runs.length) {
                const seasons = res.season_runs.map(d => d.season);
                const runs = res.season_runs.map(d => d.runs);
                Plotly.newPlot('bat-season-chart', [{
                    x: seasons, y: runs, type: 'bar',
                    marker: { color: runs, colorscale: 'Plasma' },
                    text: runs, textposition: 'outside',
                    hovertemplate: 'Season: %{x}<br>Runs: %{y}<extra></extra>',
                }], App.plotlyLayout({ showlegend: false, yaxis: { title: 'Runs' } }), App.plotlyConfig);
            }

            // Top batsmen chart
            if (topRes.players && topRes.players.length) {
                const players = topRes.players.map(p => p.batter).reverse();
                const sr = topRes.players.map(p => p.strike_rate).reverse();
                Plotly.newPlot('bat-top-chart', [{
                    x: sr, y: players, type: 'bar', orientation: 'h',
                    marker: { color: sr, colorscale: 'Viridis' },
                    hovertemplate: '%{y}: %{x}<extra></extra>',
                }], App.plotlyLayout({ showlegend: false, xaxis: { title: 'Strike Rate' } }), App.plotlyConfig);
            }
        } catch (e) {
            area.innerHTML = `<div class="empty-state"><p>Failed to load: ${e.message}</p></div>`;
        }
    },

    async loadBowler(name) {
        this.bowlerLoaded = true;
        const area = document.getElementById('bowl-stats-area');
        area.innerHTML = '<div class="spinner"></div>';
        try {
            const res = await fetch(`/api/player-stats/bowler/${encodeURIComponent(name)}`).then(r => r.json());
            if (res.error) throw new Error(res.error);
            const s = res.stats;
            const topRes = await fetch('/api/player-stats/top-bowlers?metric=wickets&min_balls=300&n=10').then(r => r.json());

            area.innerHTML = `
                <div class="kpi-grid">
                    <div class="kpi-card"><div class="kpi-value">${Math.round(s.wickets)}</div><div class="kpi-label">Wickets</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.economy}</div><div class="kpi-label">Economy</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.overs}</div><div class="kpi-label">Overs</div></div>
                    <div class="kpi-card"><div class="kpi-value">${Math.round(s.runs_conceded).toLocaleString()}</div><div class="kpi-label">Runs Conceded</div></div>
                    <div class="kpi-card"><div class="kpi-value">${Math.round(s.dot_balls)}</div><div class="kpi-label">Dot Balls</div></div>
                    <div class="kpi-card"><div class="kpi-value">${s.dot_ball_pct}%</div><div class="kpi-label">Dot Ball %</div></div>
                </div>
                <div class="grid-2">
                    <div class="card"><h3 class="section-title">📈 Wickets Per Season</h3><div class="chart-container" id="bowl-season-chart"></div></div>
                    <div class="card"><h3 class="section-title">🏆 Top 10 by Wickets</h3><div class="chart-container" id="bowl-top-chart"></div></div>
                </div>
            `;

            if (res.season_wickets && res.season_wickets.length) {
                const seasons = res.season_wickets.map(d => d.season);
                const wkts = res.season_wickets.map(d => d.wickets);
                Plotly.newPlot('bowl-season-chart', [{
                    x: seasons, y: wkts, type: 'bar',
                    marker: { color: wkts, colorscale: 'Magma' },
                    text: wkts, textposition: 'outside',
                    hovertemplate: 'Season: %{x}<br>Wickets: %{y}<extra></extra>',
                }], App.plotlyLayout({ showlegend: false, yaxis: { title: 'Wickets' } }), App.plotlyConfig);
            }

            if (topRes.players && topRes.players.length) {
                const players = topRes.players.map(p => p.bowler).reverse();
                const wk = topRes.players.map(p => p.wickets).reverse();
                Plotly.newPlot('bowl-top-chart', [{
                    x: wk, y: players, type: 'bar', orientation: 'h',
                    marker: { color: wk, colorscale: 'Cividis' },
                    hovertemplate: '%{y}: %{x} wickets<extra></extra>',
                }], App.plotlyLayout({ showlegend: false, xaxis: { title: 'Wickets' } }), App.plotlyConfig);
            }
        } catch (e) {
            area.innerHTML = `<div class="empty-state"><p>Failed to load: ${e.message}</p></div>`;
        }
    },

    async loadClusters() {
        const area = document.getElementById('cluster-area');
        area.innerHTML = '<div class="spinner"></div>';
        try {
            const res = await fetch('/api/player-clusters').then(r => r.json());
            let html = '';

            if (res.batsmen) {
                html += `<h3 class="section-title">🏏 Batsman Clusters (Silhouette: ${res.batsmen.silhouette_score})</h3><div class="cluster-grid">`;
                const groups = {};
                res.batsmen.players.forEach(p => {
                    if (!groups[p.cluster_label]) groups[p.cluster_label] = [];
                    groups[p.cluster_label].push(p.player);
                });
                for (const [label, players] of Object.entries(groups)) {
                    const badgeClass = label.toLowerCase().replace(/[\s-]/g, '-');
                    html += `<div class="cluster-card"><div class="cluster-header"><span class="cluster-badge ${badgeClass}">${label}</span><span class="cluster-count">${players.length} players</span></div><div class="cluster-players">${players.slice(0, 15).map(p => `<span class="player-tag">${p}</span>`).join('')}${players.length > 15 ? `<span class="player-tag">+${players.length - 15} more</span>` : ''}</div></div>`;
                }
                html += '</div>';
            }

            if (res.bowlers) {
                html += `<h3 class="section-title" style="margin-top:2rem">🎳 Bowler Clusters (Silhouette: ${res.bowlers.silhouette_score})</h3><div class="cluster-grid">`;
                const groups = {};
                res.bowlers.players.forEach(p => {
                    if (!groups[p.cluster_label]) groups[p.cluster_label] = [];
                    groups[p.cluster_label].push(p.player);
                });
                for (const [label, players] of Object.entries(groups)) {
                    const badgeClass = label.toLowerCase().replace(/[\s-]/g, '-');
                    html += `<div class="cluster-card"><div class="cluster-header"><span class="cluster-badge ${badgeClass}">${label}</span><span class="cluster-count">${players.length} players</span></div><div class="cluster-players">${players.slice(0, 15).map(p => `<span class="player-tag">${p}</span>`).join('')}${players.length > 15 ? `<span class="player-tag">+${players.length - 15} more</span>` : ''}</div></div>`;
                }
                html += '</div>';
            }

            area.innerHTML = html || '<div class="empty-state"><p>No cluster data available. Train models first.</p></div>';
        } catch (e) {
            area.innerHTML = `<div class="empty-state"><p>Failed to load clusters: ${e.message}</p></div>`;
        }
    },
};
