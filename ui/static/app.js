/* TradingBot Autoresearch — Alpine.js + Chart.js front-end logic. */

(() => {
  Chart.defaults.color = '#a1a1aa';
  Chart.defaults.borderColor = 'rgba(63,63,74,0.4)';
  Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, Menlo, monospace";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = '#18181c';
  Chart.defaults.plugins.tooltip.borderColor = '#3f3f4a';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
})();

function app() {
  return {
    // ───── state ────────────────────────────────────────────
    summary: { best_sharpe: 0, total: 0, keeps: 0, discards: 0, crashes: 0, keep_rate: 0, latest: null },
    results: [],
    setup: { checks: {}, data_files: [] },
    strategySource: '',
    gitLog: [],
    equity: { metrics: {}, timestamps: [], equity: [], buy_and_hold: [], drawdown: [] },
    equityLoading: false,

    loop: { running: false, lines: 0 },
    consoleLines: [],
    consoleOpen: true,
    consoleId: 0,
    sse: null,

    showLaunch: false,
    launchIters: 5,

    sortKey: '',
    sortDir: 'desc',
    filter: ['keep', 'discard', 'crash'],

    charts: { equity: null, progression: null, drawdown: null },

    // ───── kpi cards (computed) ─────────────────────────────
    get kpiCards() {
      const total = this.summary.total || 0;
      const keeps = this.summary.keeps || 0;
      const rate = total ? (keeps / total * 100).toFixed(0) : '—';
      return [
        { label: 'Best Sharpe', value: (this.summary.best_sharpe ?? 0).toFixed(4), color: 'text-emerald-400' },
        { label: 'Experiments', value: total, sub: `${keeps} kept · ${this.summary.discards} discarded · ${this.summary.crashes} crashed` },
        { label: 'Keep rate', value: total ? `${rate}%` : '—' },
        { label: 'Status', value: this.loop.running ? 'running' : 'idle', color: this.loop.running ? 'text-accent-run' : 'text-ink-100' },
      ];
    },

    get keepCount() {
      return this.results.filter(r => r.status === 'keep').length;
    },

    get filteredResults() {
      let rows = this.results.filter(r => this.filter.includes(r.status));
      if (this.sortKey) {
        const dir = this.sortDir === 'asc' ? 1 : -1;
        rows = [...rows].sort((a, b) => {
          const av = a[this.sortKey], bv = b[this.sortKey];
          if (av == null) return 1; if (bv == null) return -1;
          return (av > bv ? 1 : av < bv ? -1 : 0) * dir;
        });
      } else {
        rows = [...rows].reverse();
      }
      return rows;
    },

    // ───── lifecycle ────────────────────────────────────────
    async init() {
      await Promise.all([this.fetchSummary(), this.fetchResults(), this.fetchSetup(), this.fetchStrategy(), this.fetchGitLog()]);
      this.reloadEquity();
      this.connectSSE();
      setInterval(() => { this.fetchSummary(); this.fetchResults(); }, 4000);
      setInterval(() => { this.fetchSetup(); }, 15000);
      this.$watch('strategySource', () => this.highlightStrategy());
      this.$watch('results', () => this.renderProgression());
    },

    // ───── fetchers ─────────────────────────────────────────
    async fetchSummary() {
      try { this.summary = await fetch('/api/summary').then(r => r.json()); } catch {}
    },
    async fetchResults() {
      try { this.results = await fetch('/api/results').then(r => r.json()); } catch {}
    },
    async fetchSetup() {
      try { this.setup = await fetch('/api/setup').then(r => r.json()); } catch {}
    },
    async fetchStrategy() {
      try {
        const r = await fetch('/api/strategy').then(r => r.json());
        this.strategySource = r.source || '';
      } catch {}
    },
    async fetchGitLog() {
      try {
        const r = await fetch('/api/git-log?n=30').then(r => r.json());
        this.gitLog = r.commits || [];
      } catch {}
    },

    async reloadEquity() {
      this.equityLoading = true;
      try {
        const r = await fetch('/api/equity').then(r => r.json());
        if (r.error) throw new Error(r.error);
        this.equity = r;
        this.renderEquity();
        this.renderDrawdown();
      } catch (e) {
        console.warn('equity fetch failed:', e);
      } finally {
        this.equityLoading = false;
      }
    },

    // ───── interactions ─────────────────────────────────────
    sortBy(key) {
      if (this.sortKey === key) {
        this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.sortKey = key;
        this.sortDir = 'desc';
      }
    },

    toggleFilter(s) {
      if (this.filter.includes(s)) this.filter = this.filter.filter(x => x !== s);
      else this.filter = [...this.filter, s];
    },

    async runBacktest() {
      this.pushConsole('[ui] running backtest…');
      try {
        const r = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ window: 'val' }),
        }).then(r => r.json());
        for (const line of (r.stdout || '').split('\n')) if (line.trim()) this.pushConsole(line);
        await this.reloadEquity();
      } catch (e) {
        this.pushConsole('[ui] backtest failed: ' + e);
      }
    },

    async startLoop() {
      try {
        await fetch('/api/loop/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ iters: this.launchIters }),
        }).then(r => { if (!r.ok) throw new Error('failed to start'); });
        this.consoleOpen = true;
      } catch (e) {
        this.pushConsole('[ui] start failed: ' + e);
      }
    },

    async stopLoop() {
      try { await fetch('/api/loop/stop', { method: 'POST' }); }
      catch (e) { this.pushConsole('[ui] stop failed: ' + e); }
    },

    pushConsole(text) {
      this.consoleLines.push({ id: this.consoleId++, text });
      if (this.consoleLines.length > 1000) this.consoleLines = this.consoleLines.slice(-800);
      this.$nextTick(() => {
        const el = this.$refs.console;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    lineClass(text) {
      if (!text) return '';
      if (text.includes('KEEP')) return 'text-accent-keep';
      if (text.includes('discard') || text.includes('regression')) return 'text-accent-discard';
      if (text.includes('crash') || text.includes('ERROR') || text.includes('Traceback')) return 'text-accent-crash';
      if (text.startsWith('[loop]')) return 'text-ink-200';
      if (text.startsWith('[ui]')) return 'text-accent-run';
      return 'text-ink-300';
    },

    // ───── SSE ──────────────────────────────────────────────
    connectSSE() {
      try {
        this.sse = new EventSource('/api/loop/stream');
        this.sse.addEventListener('line', e => {
          const { line } = JSON.parse(e.data);
          this.pushConsole(line);
          if (line.startsWith('[loop] KEEP')) {
            this.fetchSummary(); this.fetchResults(); this.fetchStrategy(); this.fetchGitLog();
            this.reloadEquity();
          }
        });
        this.sse.addEventListener('status', e => {
          const s = JSON.parse(e.data);
          if (this.loop.running !== s.running) {
            this.loop = s;
            if (!s.running) {
              this.fetchSummary(); this.fetchResults(); this.fetchStrategy(); this.fetchGitLog();
              this.reloadEquity();
            }
          } else {
            this.loop = s;
          }
        });
      } catch (e) {
        console.warn('SSE connect failed:', e);
      }
    },

    // ───── charts ───────────────────────────────────────────
    renderEquity() {
      const ctx = document.getElementById('chart-equity');
      if (!ctx) return;
      if (this.charts.equity) this.charts.equity.destroy();
      const labels = this.equity.timestamps || [];
      this.charts.equity = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Strategy',
              data: this.equity.equity || [],
              borderColor: '#10b981',
              backgroundColor: 'rgba(16,185,129,0.08)',
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0.05,
            },
            {
              label: 'Buy & Hold',
              data: this.equity.buy_and_hold || [],
              borderColor: '#71717a',
              borderWidth: 1.5,
              borderDash: [4, 4],
              pointRadius: 0,
              fill: false,
              tension: 0.05,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { ticks: { maxTicksLimit: 8, callback: (v, i) => labels[i]?.slice(0, 10) || '' }, grid: { display: false } },
            y: { ticks: { callback: v => '$' + (v / 1e6).toFixed(2) + 'M' } },
          },
          plugins: { legend: { display: true, position: 'bottom' } },
        },
      });
    },

    renderDrawdown() {
      const ctx = document.getElementById('chart-drawdown');
      if (!ctx) return;
      if (this.charts.drawdown) this.charts.drawdown.destroy();
      this.charts.drawdown = new Chart(ctx, {
        type: 'line',
        data: {
          labels: this.equity.timestamps || [],
          datasets: [{
            label: 'Drawdown %',
            data: this.equity.drawdown || [],
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239,68,68,0.15)',
            borderWidth: 1,
            pointRadius: 0,
            fill: true,
            tension: 0.05,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { display: false },
            y: { ticks: { callback: v => v.toFixed(0) + '%' }, suggestedMin: -40, suggestedMax: 0, grid: { color: 'rgba(63,63,74,0.3)' } },
          },
        },
      });
    },

    renderProgression() {
      const ctx = document.getElementById('chart-progression');
      if (!ctx) return;
      const keeps = this.results.filter(r => r.status === 'keep');
      let best = 0;
      const data = keeps.map((r, i) => {
        best = Math.max(best, r.val_sharpe || 0);
        return { x: i + 1, sharpe: r.val_sharpe, best };
      });
      if (this.charts.progression) this.charts.progression.destroy();
      this.charts.progression = new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.map(d => '#' + d.x),
          datasets: [
            {
              label: 'Sharpe',
              data: data.map(d => d.sharpe),
              borderColor: '#3b82f6',
              backgroundColor: 'transparent',
              borderWidth: 1.5,
              pointRadius: 3,
              pointBackgroundColor: '#3b82f6',
              tension: 0,
            },
            {
              label: 'Best so far',
              data: data.map(d => d.best),
              borderColor: '#10b981',
              backgroundColor: 'rgba(16,185,129,0.06)',
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0,
              stepped: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { grid: { display: false } },
            y: { ticks: { callback: v => v.toFixed(2) } },
          },
          plugins: { legend: { display: true, position: 'bottom' } },
        },
      });
    },

    highlightStrategy() {
      this.$nextTick(() => {
        if (window.Prism) {
          document.querySelectorAll('code.language-python').forEach(el => Prism.highlightElement(el));
        }
      });
    },
  };
}

window.app = app;
