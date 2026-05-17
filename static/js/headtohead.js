/**
 * IPL Analytics — Head-to-Head Team Comparison Page
 */

const HeadToHead = {
    initialized: false,

    init() {
        if (!this.initialized) this.render();
        this.initialized = true;
    },

    render() {
        const page = document.getElementById('page-headtohead');
        page.innerHTML = `
            <h1 class="page-title">⚔️ Head-to-Head Comparison</h1>
            <p class="page-subtitle">Win/loss records and top performers across historical encounters</p>
            <div class="card" style="margin-bottom:2rem">
                <div class="grid-2" style="max-width:700px">
                    <div class="form-group">
                        <label class="form-label">Team A</label>
                        <select class="form-select" id="h2h-team-a"></select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Team B</label>
                        <select class="form-select" id="h2h-team-b"></select>
                    </div>
                </div>
                <button class="btn btn-primary" id="h2h-compare-btn" style="margin-top:0.5rem">⚔️ Compare</button>
            </div>
            <div id="h2h-results"></div>
        `;
        this.setupControls();
    },

    setupControls() {
        const teamA = document.getElementById('h2h-team-a');
        const teamB = document.getElementById('h2h-team-b');
        App.populateSelect(teamA, App.teams, 'Select Team A');
        App.populateSelect(teamB, App.teams, 'Select Team B');

        if (App.teams.length >= 2) {
            teamA.value = App.teams.find(t => t.includes('Mumbai')) || App.teams[0];
            teamB.value = App.teams.find(t => t.includes('Chennai')) || App.teams[1];
            this.loadComparison();
        }

        document.getElementById('h2h-compare-btn').addEventListener('click', () => this.loadComparison());
    },

    async loadComparison() {
        const teamA = document.getElementById('h2h-team-a').value;
        const teamB = document.getElementById('h2h-team-b').value;
        if (!teamA || !teamB || teamA === teamB) {
            App.toast('Please select two different teams', 'error');
            return;
        }

        const area = document.getElementById('h2h-results');
        area.innerHTML = '<div class="spinner"></div>';

        try {
            const res = await fetch(`/api/head-to-head?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`).then(r => r.json());
            if (res.error) throw new Error(res.error);

            const r = res.record;
            const perf = res.top_performers || {};

            area.innerHTML = `
                <div class="card" style="margin-bottom:1.5rem">
                    <div class="h2h-score">
                        <div class="h2h-team">
                            <div class="h2h-team-name">${teamA}</div>
                            <div class="h2h-wins team-a">${r.team_a_wins}</div>
                        </div>
                        <div>
                            <div class="h2h-vs">VS</div>
                            <div class="h2h-total">${r.total} matches</div>
                        </div>
                        <div class="h2h-team">
                            <div class="h2h-team-name">${teamB}</div>
                            <div class="h2h-wins team-b">${r.team_b_wins}</div>
                        </div>
                    </div>
                    ${r.no_result > 0 ? `<p style="text-align:center;color:var(--text-muted);font-size:0.85rem">No Result: ${r.no_result}</p>` : ''}
                    <div class="chart-container" id="h2h-pie-chart" style="min-height:280px"></div>
                </div>
                <div class="grid-2">
                    <div class="card">
                        <h3 class="section-title">🏏 Top Batsmen</h3>
                        ${perf.top_batsmen && perf.top_batsmen.length ? `
                            <table class="data-table">
                                <thead><tr><th>Batter</th><th>Runs</th></tr></thead>
                                <tbody>${perf.top_batsmen.map((p, i) => `<tr><td><span style="color:var(--accent-1);font-weight:700;margin-right:0.5rem">#${i + 1}</span>${p.batter}</td><td style="font-weight:600">${p.runs}</td></tr>`).join('')}</tbody>
                            </table>
                        ` : '<div class="empty-state"><p>No data</p></div>'}
                    </div>
                    <div class="card">
                        <h3 class="section-title">🎳 Top Bowlers</h3>
                        ${perf.top_bowlers && perf.top_bowlers.length ? `
                            <table class="data-table">
                                <thead><tr><th>Bowler</th><th>Wickets</th></tr></thead>
                                <tbody>${perf.top_bowlers.map((p, i) => `<tr><td><span style="color:var(--accent-2);font-weight:700;margin-right:0.5rem">#${i + 1}</span>${p.bowler}</td><td style="font-weight:600">${p.wickets}</td></tr>`).join('')}</tbody>
                            </table>
                        ` : '<div class="empty-state"><p>No data</p></div>'}
                    </div>
                </div>
            `;

            // Pie chart
            Plotly.newPlot('h2h-pie-chart', [{
                labels: [teamA, teamB, ...(r.no_result > 0 ? ['No Result'] : [])],
                values: [r.team_a_wins, r.team_b_wins, ...(r.no_result > 0 ? [r.no_result] : [])],
                type: 'pie', hole: 0.5,
                marker: { colors: ['#667eea', '#764ba2', ...(r.no_result > 0 ? ['#424242'] : [])] },
                textinfo: 'percent+value',
                hovertemplate: '%{label}<br>Wins: %{value}<br>%{percent}<extra></extra>',
            }], App.plotlyLayout({ showlegend: true, legend: { font: { size: 11 } }, height: 280 }), App.plotlyConfig);

        } catch (e) {
            area.innerHTML = `<div class="empty-state"><p>Failed to load: ${e.message}</p></div>`;
        }
    },
};
