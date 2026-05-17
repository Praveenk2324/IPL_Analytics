/**
 * IPL Analytics — Main Application Router & Utilities
 */

const App = {
    currentPage: 'simulator',
    teams: [],
    players: { batsmen: [], bowlers: [] },

    async init() {
        this.setupNavigation();
        this.setupMobileMenu();
        await this.loadBaseData();
        this.checkModelStatus();
        this.navigateTo('simulator');
    },

    setupNavigation() {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                this.navigateTo(page);
            });
        });
    },

    setupMobileMenu() {
        const toggle = document.getElementById('mobile-toggle');
        const sidebar = document.getElementById('sidebar');
        if (toggle) {
            toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
            document.getElementById('main-content').addEventListener('click', () => sidebar.classList.remove('open'));
        }
    },

    navigateTo(page) {
        this.currentPage = page;
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        const pageEl = document.getElementById(`page-${page}`);
        const navEl = document.getElementById(`nav-${page}`);
        if (pageEl) pageEl.classList.add('active');
        if (navEl) navEl.classList.add('active');

        // Initialize page content
        switch (page) {
            case 'simulator': Simulator.init(); break;
            case 'players': Players.init(); break;
            case 'venues': Venues.init(); break;
            case 'headtohead': HeadToHead.init(); break;
        }
        document.getElementById('sidebar').classList.remove('open');
    },

    async loadBaseData() {
        try {
            const [teamsRes, playersRes] = await Promise.all([
                fetch('/api/teams').then(r => r.json()),
                fetch('/api/players').then(r => r.json()),
            ]);
            this.teams = teamsRes.teams || [];
            this.players = { batsmen: playersRes.batsmen || [], bowlers: playersRes.bowlers || [] };
        } catch (e) {
            this.toast('Failed to load base data', 'error');
        }
    },

    async checkModelStatus() {
        try {
            const res = await fetch('/api/model-info').then(r => r.json());
            const loaded = Object.values(res).filter(m => m.status === 'loaded').length;
            const total = Object.keys(res).length;
            const dot = document.querySelector('.status-dot');
            const text = document.querySelector('.status-text');
            if (loaded >= 2) {
                dot.classList.add('ready');
                text.textContent = `${loaded}/${total} models ready`;
            } else {
                text.textContent = `${loaded}/${total} models loaded`;
            }
        } catch (e) { /* silent */ }
    },

    toast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    },

    populateSelect(selectEl, options, placeholder = 'Select...') {
        selectEl.innerHTML = `<option value="">${placeholder}</option>` +
            options.map(o => `<option value="${o}">${o}</option>`).join('');
    },

    plotlyLayout(extra = {}) {
        return {
            template: 'plotly_dark',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter', color: '#e8eaf6', size: 12 },
            margin: { l: 50, r: 30, t: 40, b: 50 },
            ...extra,
        };
    },

    plotlyConfig: { displayModeBar: false, responsive: true },
};

document.addEventListener('DOMContentLoaded', () => App.init());
