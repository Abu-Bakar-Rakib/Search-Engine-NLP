"""Modern Web Search Interface for lite_search."""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>lite_search | Hybrid E-Commerce Search</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.25);
      --accent-hover: #4f46e5;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --font: 'Plus Jakarta Sans', sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 25%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 50% 85%, rgba(244, 63, 94, 0.06) 0%, transparent 50%);
      background-attachment: fixed;
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem;
      width: 100%;
    }

    /* Header */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 2rem;
    }
    .logo-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo-badge {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: grid;
      place-items: center;
      box-shadow: 0 4px 14px var(--accent-glow);
    }
    .logo-badge svg { width: 22px; height: 22px; fill: white; }
    .brand-title {
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-tag {
      font-size: 0.75rem;
      padding: 0.2rem 0.55rem;
      background: rgba(99, 102, 241, 0.15);
      border: 1px solid rgba(99, 102, 241, 0.3);
      color: #a5b4fc;
      border-radius: 9999px;
      font-weight: 600;
    }
    .api-link {
      font-size: 0.85rem;
      color: var(--text-muted);
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.9rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      transition: all 0.2s;
    }
    .api-link:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
    }

    /* Search Box */
    .search-section {
      position: relative;
      margin-bottom: 1.5rem;
    }
    .search-bar-wrap {
      position: relative;
      display: flex;
      align-items: center;
      background: rgba(18, 24, 38, 0.85);
      border: 1.5px solid rgba(255, 255, 255, 0.12);
      border-radius: 16px;
      padding: 0.4rem 0.6rem 0.4rem 1.25rem;
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(16px);
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .search-bar-wrap:focus-within {
      border-color: var(--accent);
      box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.35), 0 0 0 2px var(--accent-glow);
    }
    .search-icon {
      color: var(--text-dim);
      margin-right: 0.8rem;
      flex-shrink: 0;
    }
    .search-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      font-size: 1.1rem;
      color: #fff;
      font-family: inherit;
      padding: 0.6rem 0;
    }
    .search-input::placeholder { color: var(--text-dim); }
    .search-btn {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      border: none;
      color: white;
      font-family: inherit;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
      transition: transform 0.15s, opacity 0.15s;
    }
    .search-btn:hover { opacity: 0.95; transform: translateY(-1px); }
    .search-btn:active { transform: translateY(0); }

    /* Voice Search Button */
    .voice-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
      cursor: pointer;
      width: 40px;
      height: 40px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-right: 0.6rem;
      transition: all 0.2s ease;
      position: relative;
      flex-shrink: 0;
    }
    .voice-btn:hover {
      color: #fff;
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.15);
      transform: translateY(-1px);
    }
    .voice-btn.recording {
      color: #fff;
      border-color: #f43f5e;
      background: #f43f5e;
      animation: pulse-ring 1.4s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    @keyframes pulse-ring {
      0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(244, 63, 94, 0); }
      100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
    }

    /* Voice Status Overlay */
    .voice-status-bar {
      display: none;
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: rgba(18, 24, 38, 0.96);
      border: 1px solid rgba(244, 63, 94, 0.4);
      border-radius: 12px;
      padding: 0.85rem 1.25rem;
      font-size: 0.92rem;
      color: #f1f5f9;
      backdrop-filter: blur(20px);
      z-index: 55;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 12px 36px rgba(0,0,0,0.6);
    }
    .voice-wave {
      display: flex;
      align-items: center;
      gap: 3px;
    }
    .voice-bar {
      width: 3px;
      height: 12px;
      background: #f43f5e;
      border-radius: 2px;
      animation: voice-wave-anim 1s ease-in-out infinite;
    }
    .voice-bar:nth-child(2) { animation-delay: 0.15s; height: 18px; }
    .voice-bar:nth-child(3) { animation-delay: 0.3s; height: 14px; }
    .voice-bar:nth-child(4) { animation-delay: 0.45s; height: 22px; }
    .voice-bar:nth-child(5) { animation-delay: 0.6s; height: 10px; }
    @keyframes voice-wave-anim {
      0%, 100% { transform: scaleY(0.4); }
      50% { transform: scaleY(1); }
    }
    .voice-cancel-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: var(--text-dim);
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      font-family: inherit;
      transition: all 0.15s;
    }
    .voice-cancel-btn:hover {
      background: rgba(255, 255, 255, 0.15);
      color: #fff;
    }

    /* Autocomplete dropdown */
    .suggest-dropdown {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: rgba(18, 24, 38, 0.96);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      box-shadow: 0 12px 36px rgba(0,0,0,0.6);
      backdrop-filter: blur(20px);
      z-index: 50;
      overflow: hidden;
      display: none;
    }
    .suggest-item {
      padding: 0.75rem 1.25rem;
      font-size: 0.95rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      color: var(--text-muted);
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      transition: background 0.15s, color 0.15s;
    }
    .suggest-item:last-child { border-bottom: none; }
    .suggest-item:hover {
      background: rgba(99, 102, 241, 0.15);
      color: #fff;
    }

    /* Query Presets Chips */
    .chips-label {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }
    .chips-group {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 2rem;
    }
    .chip {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
      padding: 0.35rem 0.8rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .chip:hover {
      background: rgba(99, 102, 241, 0.12);
      border-color: rgba(99, 102, 241, 0.3);
      color: #c7d2fe;
      transform: translateY(-1px);
    }

    /* Meta & Intent Banner */
    .meta-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 1.75rem;
      display: none;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .meta-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
      padding-bottom: 0.8rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
    }
    .source-rules { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .source-embedding { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
    .source-llm { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .cached-pill { background: rgba(255, 255, 255, 0.08); color: #cbd5e1; font-family: var(--mono); }
    .latency-pill { font-family: var(--mono); color: var(--text-muted); font-size: 0.8rem; }
    
    .intent-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.8rem;
    }
    .intent-item {
      font-size: 0.8rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 0.2rem 0.55rem;
      color: #94a3b8;
    }
    .intent-item span { color: #f1f5f9; font-weight: 600; }
    .notes-list {
      margin-top: 0.7rem;
      font-size: 0.82rem;
      color: var(--amber);
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    /* Results Grid */
    .results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1.25rem;
    }
    .product-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
      position: relative;
    }
    .product-card:hover {
      transform: translateY(-2px);
      border-color: rgba(99, 102, 241, 0.3);
      box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
    }
    .card-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.5rem;
      margin-bottom: 0.6rem;
    }
    .rank-tag {
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--text-dim);
    }
    .match-label {
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      background: rgba(99, 102, 241, 0.15);
      color: #c7d2fe;
    }
    .card-title {
      font-size: 1.05rem;
      font-weight: 700;
      line-height: 1.4;
      margin-bottom: 0.5rem;
      color: #fff;
    }
    .card-brand-cat {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 0.8rem;
    }
    .card-attrs {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-bottom: 1rem;
    }
    .attr-badge {
      font-size: 0.75rem;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.06);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      color: var(--text-muted);
    }
    .card-bottom {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 0.8rem;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .price-val {
      font-size: 1.25rem;
      font-weight: 800;
      color: #34d399;
      letter-spacing: -0.01em;
    }
    .btn-similar {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.35rem 0.7rem;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-similar:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: rgba(99, 102, 241, 0.4);
      color: #fff;
    }
    .score-bar {
      margin-top: 0.5rem;
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--mono);
      display: flex;
      gap: 0.6rem;
    }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 4rem 1rem;
      color: var(--text-dim);
      grid-column: 1 / -1;
    }
    .empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.5; }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-group">
        <div class="logo-badge">
          <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
        </div>
        <div>
          <span class="brand-title">lite_search</span>
          <span class="brand-tag">v0.1.0</span>
        </div>
      </div>
      <a href="/docs" target="_blank" class="api-link">
        <span>Swagger API Docs</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/></svg>
      </a>
    </header>

    <div class="search-section">
      <div class="search-bar-wrap">
        <svg class="search-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
        <input type="text" id="searchInput" class="search-input" placeholder="Search anything, e.g. 'black nike running shoe under 5000' or 'panjabi 2000 er niche'..." autofocus autocomplete="off">
        <button type="button" id="voiceBtn" class="voice-btn" title="Voice Search (Speak in Bangla or English)">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
        </button>
        <button id="searchBtn" class="search-btn">Search</button>
      </div>
      <div id="voiceStatusBar" class="voice-status-bar">
        <div style="display: flex; align-items: center; gap: 0.75rem; flex: 1;">
          <div class="voice-wave">
            <div class="voice-bar"></div>
            <div class="voice-bar"></div>
            <div class="voice-bar"></div>
            <div class="voice-bar"></div>
            <div class="voice-bar"></div>
          </div>
          <span id="voiceStatusText">Listening... Speak now</span>
        </div>
        <button type="button" onclick="cancelVoiceSearch()" class="voice-cancel-btn">Cancel</button>
      </div>
      <div id="suggestBox" class="suggest-dropdown"></div>
    </div>

    <div class="chips-label">Try these sample queries</div>
    <div class="chips-group">
      <button class="chip" onclick="runSearch('পানি গরম করার ইলেকট্রিক কেটলি')">পানি গরম করার ইলেকট্রিক কেটলি (Electric Kettle)</button>
      <button class="chip" onclick="runSearch('বিয়ের অনুষ্ঠানের জন্য শাড়ি')">বিয়ের অনুষ্ঠানের জন্য শাড়ি (Bangla)</button>
      <button class="chip" onclick="runSearch('মায়ের জন্য শাড়ি ২০০০ টাকার নিচে')">মায়ের জন্য শাড়ি ২০০০ টাকার নিচে (Bangla)</button>
      <button class="chip" onclick="runSearch('রান্নার কড়াই')">রান্নার কড়াই (Bangla)</button>
      <button class="chip" onclick="runSearch('pant')">pant</button>
      <button class="chip" onclick="runSearch('churi')">churi (knife)</button>
      <button class="chip" onclick="runSearch('black nike running shoe under 5000')">black nike running shoe under 5000</button>
      <button class="chip" onclick="runSearch('panjabi 2000 er niche')">panjabi 2000 er niche (Banglish)</button>
      <button class="chip" onclick="runSearch('cotton shirt size L for men')">cotton shirt size L for men</button>
      <button class="chip" onclick="runSearch('gift for a toddler who loves building things')">gift for toddler (semantic intent)</button>
      <button class="chip" onclick="runSearch('wireless headphones')">wireless headphones</button>
      <button class="chip" onclick="runSearch('leather wallet for men')">leather wallet for men</button>
    </div>

    <div id="metaBox" class="meta-box">
      <div class="meta-header">
        <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;">
          <span id="sourceBadge" class="badge-pill"></span>
          <span id="cacheBadge" class="badge-pill cached-pill"></span>
          <span id="countBadge" class="badge-pill" style="background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.15); color:#fff;"></span>
        </div>
        <span id="latencyBadge" class="latency-pill"></span>
      </div>
      <div id="intentTags" class="intent-tags"></div>
      <div id="notesList" class="notes-list"></div>
    </div>

    <div id="resultsGrid" class="results-grid">
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div>Enter a query above or click any preset to see hybrid search ranking in action.</div>
      </div>
    </div>
  </div>

  <script>
    const input = document.getElementById('searchInput');
    const btn = document.getElementById('searchBtn');
    const suggestBox = document.getElementById('suggestBox');
    const metaBox = document.getElementById('metaBox');
    const resultsGrid = document.getElementById('resultsGrid');
    let debounceTimer;

    input.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      const q = e.target.value.trim();
      if (q.length < 2) {
        suggestBox.style.display = 'none';
        return;
      }
      debounceTimer = setTimeout(() => {
        fetch(`/suggest?q=${encodeURIComponent(q)}&k=6`)
          .then(res => res.json())
          .then(data => {
            if (data.suggestions && data.suggestions.length) {
              suggestBox.innerHTML = data.suggestions.map(s => 
                `<div class="suggest-item" onclick="selectSuggest('${s.replace(/'/g, "\\\\'")}')">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
                  <span>${s}</span>
                </div>`
              ).join('');
              suggestBox.style.display = 'block';
            } else {
              suggestBox.style.display = 'none';
            }
          }).catch(() => { suggestBox.style.display = 'none'; });
      }, 150);
    });

    document.addEventListener('click', (e) => {
      if (!suggestBox.contains(e.target) && e.target !== input) {
        suggestBox.style.display = 'none';
      }
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        suggestBox.style.display = 'none';
        runSearch(input.value);
      }
    });
    btn.addEventListener('click', () => {
      suggestBox.style.display = 'none';
      runSearch(input.value);
    });

    function selectSuggest(q) {
      input.value = q;
      suggestBox.style.display = 'none';
      runSearch(q);
    }

    function runSearch(query) {
      if (!query || !query.trim()) return;
      input.value = query;
      btn.innerHTML = '<span class="spinner"></span>';
      
      // Request up to 100 relevant products; backend dynamically filters to only what actually matches
      fetch(`/search?q=${encodeURIComponent(query)}&k=100`)
        .then(res => res.json())
        .then(data => {
          btn.innerHTML = 'Search';
          renderResponse(data);
        })
        .catch(err => {
          btn.innerHTML = 'Search';
          resultsGrid.innerHTML = `<div class="empty-state">Error performing search: ${err}</div>`;
        });
    }

    function renderResponse(data) {
      metaBox.style.display = 'block';
      
      // Source badge
      const srcEl = document.getElementById('sourceBadge');
      srcEl.className = 'badge-pill';
      const src = (data.source || 'embedding').toLowerCase();
      if (src.includes('rules')) {
        srcEl.classList.add('source-rules');
        srcEl.innerText = '⚡ RULES TIER';
      } else if (src.includes('llm')) {
        srcEl.classList.add('source-llm');
        srcEl.innerText = '🤖 LLM TIER';
      } else {
        srcEl.classList.add('source-embedding');
        srcEl.innerText = '🧠 EMBEDDINGS + BM25';
      }

      // Cache badge & latency
      document.getElementById('cacheBadge').innerText = data.cached ? '⚡ Cached Hit' : 'Origin Computation';
      document.getElementById('latencyBadge').innerText = `${data.took_ms} ms`;
      const numResults = (data.results && data.results.length) || 0;
      document.getElementById('countBadge').innerText = `📦 ${numResults} Products Found`;

      // Intent breakdown
      const intentBox = document.getElementById('intentTags');
      const intent = data.intent || {};
      const intentKeys = Object.keys(intent);
      if (intentKeys.length) {
        intentBox.innerHTML = intentKeys.map(k => 
          `<div class="intent-item">${k}: <span>${intent[k]}</span></div>`
        ).join('');
      } else {
        intentBox.innerHTML = '<div class="intent-item">Intent: <span>Free semantic search</span></div>';
      }

      // Notes
      const notesEl = document.getElementById('notesList');
      if (data.notes && data.notes.length) {
        notesEl.innerHTML = data.notes.map(n => `<div>• ${n}</div>`).join('');
      } else {
        notesEl.innerHTML = '';
      }

      // Products grid
      if (!data.results || !data.results.length) {
        resultsGrid.innerHTML = '<div class="empty-state">No matching products found.</div>';
        return;
      }

      resultsGrid.innerHTML = data.results.map(p => `
        <div class="product-card">
          <div>
            <div class="card-top">
              <span class="rank-tag">#${p.rank} (${p.id})</span>
              ${p.matched ? `<span class="match-label">${p.matched}</span>` : ''}
            </div>
            <div class="card-title">${p.title}</div>
            <div class="card-brand-cat">${p.brand ? '<strong>' + p.brand + '</strong> • ' : ''}${p.category}</div>
            <div class="card-attrs">
              ${p.color ? `<span class="attr-badge">${p.color}</span>` : ''}
              ${p.material ? `<span class="attr-badge">${p.material}</span>` : ''}
              ${p.gender ? `<span class="attr-badge">${p.gender}</span>` : ''}
              ${p.sizes ? `<span class="attr-badge">Sizes: ${p.sizes}</span>` : ''}
            </div>
          </div>
          <div>
            <div class="card-bottom">
              <span class="price-val">৳${Number(p.price).toLocaleString()}</span>
              <button class="btn-similar" onclick="findSimilar('${p.id}')">Similar</button>
            </div>
            <div class="score-bar">
              <span>Score: ${p.score}</span>
              <span>Cosine: ${p.cosine}</span>
              <span>BM25: ${p.bm25}</span>
            </div>
          </div>
        </div>
      `).join('');
    }

    function findSimilar(id) {
      btn.innerHTML = '<span class="spinner"></span>';
      fetch(`/similar/${id}?k=8`)
        .then(res => res.json())
        .then(data => {
          btn.innerHTML = 'Search';
          metaBox.style.display = 'block';
          document.getElementById('sourceBadge').className = 'badge-pill source-embedding';
          document.getElementById('sourceBadge').innerText = '🧠 NEAREST NEIGHBOURS';
          document.getElementById('cacheBadge').innerText = 'Vector Similarity';
          document.getElementById('latencyBadge').innerText = '';
          document.getElementById('intentTags').innerHTML = `<div class="intent-item">Similar to Product: <span>${id}</span></div>`;
          document.getElementById('notesList').innerHTML = '';
          
          if (!data.results || !data.results.length) {
            resultsGrid.innerHTML = '<div class="empty-state">No similar products found.</div>';
            return;
          }
          resultsGrid.innerHTML = data.results.map(p => `
            <div class="product-card">
              <div>
                <div class="card-top">
                  <span class="rank-tag">#${p.rank} (${p.id})</span>
                  <span class="match-label">Sim: ${p.similarity}</span>
                </div>
                <div class="card-title">${p.title}</div>
                <div class="card-brand-cat">${p.brand ? '<strong>' + p.brand + '</strong> • ' : ''}${p.category}</div>
                <div class="card-attrs">
                  ${p.color ? `<span class="attr-badge">${p.color}</span>` : ''}
                  ${p.material ? `<span class="attr-badge">${p.material}</span>` : ''}
                  ${p.gender ? `<span class="attr-badge">${p.gender}</span>` : ''}
                </div>
              </div>
              <div class="card-bottom">
                <span class="price-val">৳${Number(p.price).toLocaleString()}</span>
                <button class="btn-similar" onclick="findSimilar('${p.id}')">Similar</button>
              </div>
            </div>
          `).join('');
        });
    }

    // Voice Search Implementation (Web Speech API)
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceStatusBar = document.getElementById('voiceStatusBar');
    const voiceStatusText = document.getElementById('voiceStatusText');

    let recognition = null;
    let isListening = false;
    let silenceTimer = null;
    let maxRecordingTimer = null;
    let shouldSearchOnEnd = true;
    let latestTranscript = '';

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      if (voiceBtn) {
        voiceBtn.title = 'Voice search is not supported in this browser. Please use Chrome, Edge, or Safari.';
        voiceBtn.style.opacity = '0.5';
      }
    } else {
      recognition = new SpeechRecognition();
      // Continuous mode allows capturing entire phrase with natural short pauses
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      // bn-BD natively transcribes both Bangla speech and English product terms
      recognition.lang = 'bn-BD';

      recognition.onstart = () => {
        isListening = true;
        shouldSearchOnEnd = true;
        latestTranscript = '';
        voiceBtn.classList.add('recording');
        voiceStatusBar.style.display = 'flex';
        voiceStatusText.innerText = 'Listening... Speak now';
        suggestBox.style.display = 'none';

        // Safety maximum recording window: 10 seconds
        clearTimeout(maxRecordingTimer);
        maxRecordingTimer = setTimeout(() => {
          stopVoiceSearch(true);
        }, 10000);
      };

      // Native Voice Activity Detector: triggers quickly when speech stops
      recognition.onspeechend = () => {
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
          stopVoiceSearch(true);
        }, 400);
      };

      recognition.onresult = (event) => {
        let finalPart = '';
        let interimPart = '';
        let hasFinal = false;

        for (let i = 0; i < event.results.length; ++i) {
          const res = event.results[i];
          const text = res[0].transcript;
          if (res.isFinal) {
            finalPart += text + ' ';
            hasFinal = true;
          } else {
            interimPart += text;
          }
        }

        const combined = (finalPart + interimPart).trim();
        // Clean trailing punctuation inserted by speech engine (e.g. "shoe.", "saree?")
        const cleaned = combined.replace(/[.,?!]+$/, '').trim();

        if (cleaned) {
          latestTranscript = cleaned;
          input.value = cleaned;
          voiceStatusText.innerText = cleaned;

          // Snappy silence timer: 550ms if engine finalized phrase, or 900ms for interim
          clearTimeout(silenceTimer);
          silenceTimer = setTimeout(() => {
            stopVoiceSearch(true);
          }, hasFinal ? 550 : 900);
        }
      };

      recognition.onerror = (event) => {
        clearTimeout(silenceTimer);
        clearTimeout(maxRecordingTimer);
        if (event.error === 'no-speech') {
          voiceStatusText.innerText = 'No speech detected. Click mic to try again.';
        } else if (event.error === 'not-allowed') {
          voiceStatusText.innerText = 'Microphone permission blocked. Please allow mic access.';
        } else {
          voiceStatusText.innerText = `Voice notice: ${event.error}`;
        }
        setTimeout(() => stopVoiceSearch(false), 1600);
      };

      recognition.onend = () => {
        clearTimeout(silenceTimer);
        clearTimeout(maxRecordingTimer);
        isListening = false;
        voiceBtn.classList.remove('recording');
        voiceStatusBar.style.display = 'none';

        if (shouldSearchOnEnd) {
          const query = (latestTranscript || input.value).trim().replace(/[.,?!]+$/, '');
          if (query && query.length >= 2) {
            runSearch(query);
          }
        }
      };

      voiceBtn.addEventListener('click', () => {
        if (isListening) {
          // If clicked while listening, finish and search immediately
          stopVoiceSearch(true);
        } else {
          startVoiceSearch();
        }
      });
    }

    function startVoiceSearch() {
      if (!recognition) {
        alert('Voice search is not supported in this browser. Please use Google Chrome, Microsoft Edge, or Safari.');
        return;
      }
      try {
        clearTimeout(silenceTimer);
        clearTimeout(maxRecordingTimer);
        input.value = '';
        latestTranscript = '';
        recognition.start();
      } catch (err) {
        console.warn('Voice start warning:', err);
      }
    }

    function stopVoiceSearch(andSearch = true) {
      clearTimeout(silenceTimer);
      clearTimeout(maxRecordingTimer);
      shouldSearchOnEnd = andSearch;
      if (recognition && isListening) {
        try {
          recognition.stop();
        } catch (e) {}
      }
      isListening = false;
      if (voiceBtn) voiceBtn.classList.remove('recording');
      if (voiceStatusBar) voiceStatusBar.style.display = 'none';
      if (andSearch) {
        const query = (latestTranscript || input.value).trim().replace(/[.,?!]+$/, '');
        if (query && query.length >= 2) {
          runSearch(query);
        }
      }
    }

    function cancelVoiceSearch() {
      latestTranscript = '';
      stopVoiceSearch(false);
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isListening) {
        cancelVoiceSearch();
      }
    });

    // Auto-search if ?q= parameter is present in URL
    const urlParams = new URLSearchParams(window.location.search);
    const qParam = urlParams.get('q');
    if (qParam) {
      input.value = qParam;
      runSearch(qParam);
    }
  </script>
</body>
</html>
"""
