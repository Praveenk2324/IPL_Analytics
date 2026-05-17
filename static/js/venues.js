/**
 * IPL Analytics — Venue Heatmap Page
 */

const Venues = {
    initialized: false,

    init() {
        if (!this.initialized) {
            this.render();
            this.loadData();
        }
        this.initialized = true;
    },

    render() {
        const page = document.getElementById('page-venues');
        page.innerHTML = `
            <h1 class="page-title">🏟️ Venue Heatmap</h1>
            <p class="page-subtitle">Average first-innings scores and chase vs defend win rates per ground</p>
            <div id="venue-content"><div class="spinner"></div></div>
        `;
    },

    async loadData() {
        const area = document.getElementById('venue-content');
        try {
            const res = await fetch('/api/venue-stats?min_matches=5').then(r => r.json());
            const venues = res.venues || [];
            if (!venues.length) { area.innerHTML = '<div class="empty-state"><p>No venue data</p></div>'; return; }

            area.innerHTML = `
                <div class="grid-2" style="margin-bottom:2rem">
                    <div class="card"><h3 class="section-title">📊 Avg First Innings Score</h3><div class="chart-container" id="venue-avg-chart"></div></div>
                    <div class="card"><h3 class="section-title">🏆 Chase vs Defend Win %</h3><div class="chart-container" id="venue-chase-chart"></div></div>
                </div>
                <div class="card">
                    <h3 class="section-title">📋 Venue Details</h3>
                    <div style="overflow-x:auto">
                        <table class="data-table" id="venue-table">
                            <thead>
                                <tr>
                                    <th>Venue</th><th>Matches</th><th>Avg Score</th>
                                    <th>Chase Win %</th><th>Defend Win %</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>
            `;

            // Top 20 venues by matches
            const top = venues.sort((a, b) => b.total_matches - a.total_matches).slice(0, 20);
            const sorted = [...top].sort((a, b) => b.avg_score - a.avg_score);

            // Avg score chart
            const vNames = sorted.map(v => v.venue.length > 35 ? v.venue.substring(0, 32) + '...' : v.venue).reverse();
            const scores = sorted.map(v => v.avg_score).reverse();
            const colors = scores.map(s => {
                const norm = (s - Math.min(...scores)) / (Math.max(...scores) - Math.min(...scores) + 0.1);
                return `hsl(${(1 - norm) * 240 + norm * 0}, 70%, 55%)`;
            });

            Plotly.newPlot('venue-avg-chart', [{
                x: scores, y: vNames, type: 'bar', orientation: 'h',
                marker: { color: colors },
                text: scores, textposition: 'outside', textfont: { size: 10 },
                hovertemplate: '%{y}<br>Avg Score: %{x}<extra></extra>',
            }], App.plotlyLayout({ height: Math.max(400, top.length * 28), margin: { l: 220 }, showlegend: false }), App.plotlyConfig);

            // Chase vs Defend chart
            const cvNames = sorted.map(v => v.venue.length > 30 ? v.venue.substring(0, 27) + '...' : v.venue).reverse();
            Plotly.newPlot('venue-chase-chart', [
                { x: sorted.map(v => v.chase_win_pct).reverse(), y: cvNames, type: 'bar', orientation: 'h', name: 'Chase', marker: { color: 'rgba(76,175,80,0.7)' }, hovertemplate: '%{y}<br>Chase: %{x}%<extra></extra>' },
                { x: sorted.map(v => v.defend_win_pct).reverse(), y: cvNames, type: 'bar', orientation: 'h', name: 'Defend', marker: { color: 'rgba(239,83,80,0.7)' }, hovertemplate: '%{y}<br>Defend: %{x}%<extra></extra>' },
            ], App.plotlyLayout({ height: Math.max(400, top.length * 28), margin: { l: 200 }, barmode: 'group', legend: { x: 0.7, y: 1.05, orientation: 'h' } }), App.plotlyConfig);

            // Table
            const tbody = document.querySelector('#venue-table tbody');
            venues.sort((a, b) => b.total_matches - a.total_matches).forEach(v => {
                const chaseColor = v.chase_win_pct > 50 ? 'var(--green)' : 'var(--text-secondary)';
                const defendColor = v.defend_win_pct > 50 ? 'var(--red)' : 'var(--text-secondary)';
                tbody.innerHTML += `<tr>
                    <td style="font-weight:500">${v.venue}</td>
                    <td>${v.total_matches}</td>
                    <td style="font-weight:600">${v.avg_score}</td>
                    <td style="color:${chaseColor};font-weight:600">${v.chase_win_pct}%</td>
                    <td style="color:${defendColor};font-weight:600">${v.defend_win_pct}%</td>
                </tr>`;
            });

        } catch (e) {
            area.innerHTML = `<div class="empty-state"><p>Failed to load venue data: ${e.message}</p></div>`;
        }
    },
};
