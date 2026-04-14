/* ================================================================
   Design Explorer — 个人内容站 设计决策树系统
   ================================================================ */

const $ = (s) => document.querySelector(s);

function wrapPage(css, body) {
  return `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}html{scroll-behavior:smooth;}img{max-width:100%;display:block;}a{color:inherit;text-decoration:none;}${css}</style></head><body>${body}</body></html>`;
}

// ============================== VARIANTS ==============================

const variants = {

// ──────────────────────── 1. 数字花园 (Digital Garden) ────────────────────────
garden: {
  name: '数字花园',
  subtitle: '非线性知识网络，随意探索，没有首页只有入口',
  tags: ['garden', 'networked', 'explore'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#f7f6f3;color:#37352f;min-height:100vh;}
    .topbar{padding:20px 32px;display:flex;justify-content:space-between;align-items:center;max-width:880px;margin:0 auto;}
    .topbar-name{font-size:14px;font-weight:600;color:#37352f;}
    .topbar-links{display:flex;gap:20px;font-size:13px;color:#999;}
    .topbar-links a:hover{color:#37352f;}
    .garden-hero{max-width:880px;margin:0 auto;padding:80px 32px 48px;}
    .garden-hero h1{font-size:32px;font-weight:700;letter-spacing:-0.02em;line-height:1.3;}
    .garden-hero p{font-size:15px;color:#787774;margin-top:12px;line-height:1.7;max-width:520px;}
    .garden-search{margin-top:24px;padding:12px 16px;width:100%;max-width:400px;border:1px solid #e3e2de;border-radius:8px;font-size:14px;background:#fff;color:#37352f;outline:none;}
    .garden-search:focus{border-color:#37352f;}
    .garden-stats{display:flex;gap:24px;margin-top:20px;font-size:12px;color:#b4b4b0;}
    .garden-stats span strong{color:#787774;font-weight:600;}
    .section-label{max-width:880px;margin:0 auto;padding:0 32px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#b4b4b0;margin-bottom:12px;}
    .notes-grid{max-width:880px;margin:0 auto;padding:0 32px 48px;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
    .note-card{padding:20px;background:#fff;border-radius:8px;border:1px solid #e8e8e4;cursor:pointer;transition:all 0.15s;}
    .note-card:hover{border-color:#c8c8c4;transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,0.04);}
    .note-card h3{font-size:14px;font-weight:600;margin-bottom:6px;line-height:1.4;}
    .note-card p{font-size:12px;color:#999;line-height:1.6;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
    .note-card .meta{display:flex;gap:8px;margin-top:10px;}
    .note-card .pill{font-size:10px;padding:2px 8px;border-radius:10px;background:#f0efec;color:#787774;}
    .note-card .pill.seedling{background:#e8f5e9;color:#4caf50;}
    .note-card .pill.evergreen{background:#e3f2fd;color:#2196f3;}
    .note-card .pill.budding{background:#fff8e1;color:#ff8f00;}
    .graph-hint{max-width:880px;margin:0 auto;padding:16px 32px 48px;display:flex;align-items:center;gap:12px;font-size:12px;color:#b4b4b0;border-top:1px solid #eee;margin-top:12px;}
    .graph-dot{width:8px;height:8px;border-radius:50%;background:#c8c8c4;}
    .footer{max-width:880px;margin:0 auto;padding:24px 32px;font-size:11px;color:#c8c8c4;}
  `, `
    <div class="topbar">
      <span class="topbar-name">🌱 夏天的花园</span>
      <div class="topbar-links"><a href="#">随机漫步</a><a href="#">图谱</a><a href="#">关于</a></div>
    </div>
    <section class="garden-hero">
      <h1>欢迎来到我的数字花园</h1>
      <p>这里不是博客——没有发布日期，没有完美的文章。只有不同成熟度的想法，像植物一样生长。你可以随意探索。</p>
      <input class="garden-search" placeholder="搜索笔记…" />
      <div class="garden-stats"><span><strong>42</strong> 篇笔记</span><span><strong>7</strong> 个主题</span><span><strong>128</strong> 条链接</span></div>
    </section>
    <div class="section-label">最近培育</div>
    <div class="notes-grid">
      <div class="note-card"><h3>系统设计的第一性原理</h3><p>从最基本的约束出发，推导出合理的架构决策。不是最佳实践的堆砌，而是理解为什么。</p><div class="meta"><span class="pill evergreen">🌳 常青</span><span class="pill">系统设计</span></div></div>
      <div class="note-card"><h3>Rust 所有权与人生责任</h3><p>学 Rust 第 100 天的感悟。所有权模型教会我的不只是内存管理。</p><div class="meta"><span class="pill budding">🌿 成长中</span><span class="pill">Rust</span></div></div>
      <div class="note-card"><h3>为什么好的 API 像好的散文</h3><p>命名即设计。一个好的函数名应该让读者忘记实现细节。</p><div class="meta"><span class="pill evergreen">🌳 常青</span><span class="pill">设计</span></div></div>
      <div class="note-card"><h3>厨房里的系统思维</h3><p>做菜和写代码的共同点：预处理、并行、时间管理、错误恢复。</p><div class="meta"><span class="pill seedling">🌱 萌芽</span><span class="pill">生活</span></div></div>
      <div class="note-card"><h3>静态网站的文艺复兴</h3><p>在 SPA 统治的时代，为什么越来越多人回到了静态生成？</p><div class="meta"><span class="pill budding">🌿 成长中</span><span class="pill">前端</span></div></div>
      <div class="note-card"><h3>重读《禅与摩托车维修艺术》</h3><p>Quality 不是一个形容词，而是一个事件——它发生在你和世界之间。</p><div class="meta"><span class="pill seedling">🌱 萌芽</span><span class="pill">阅读</span></div></div>
    </div>
    <div class="graph-hint"><span class="graph-dot"></span>提示：每篇笔记底部都有相关链接，点击即可跳转。你也可以通过「图谱」查看知识网络全貌。</div>
    <footer class="footer">这座花园从 2024 年开始种植，使用 Astro 构建</footer>
  `),
},

// ──────────────────────── 2. 手帐日记 (Journal) ────────────────────────
journal: {
  name: '手帐日记',
  subtitle: '时间线叙事，温暖私人化，像翻开一本笔记本',
  tags: ['journal', 'warm', 'timeline'],
  html: () => wrapPage(`
    body{font-family:'Lora','Noto Serif SC',Georgia,serif;background:#faf7f2;color:#3d3226;min-height:100vh;}
    .j-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 40px;max-width:720px;margin:0 auto;}
    .j-nav-brand{font-family:'DM Serif Display',serif;font-size:20px;color:#3d3226;}
    .j-nav-links{display:flex;gap:20px;font-family:'Inter',sans-serif;font-size:12px;color:#b8a898;}
    .j-hero{max-width:720px;margin:0 auto;padding:80px 40px 48px;}
    .j-hero h1{font-family:'DM Serif Display',serif;font-size:36px;font-weight:400;line-height:1.35;}
    .j-hero h1 em{font-style:italic;color:#c4956a;}
    .j-hero .intro{font-family:'Inter',sans-serif;font-size:14px;color:#a09486;margin-top:16px;line-height:1.8;max-width:480px;}
    .j-divider{max-width:720px;margin:0 auto;padding:0 40px;}
    .j-divider hr{border:none;border-top:1px solid #ede6dd;margin:0;}
    .timeline{max-width:720px;margin:0 auto;padding:32px 40px;}
    .t-month{font-family:'Inter',sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:0.12em;color:#c4b8aa;margin-bottom:20px;margin-top:32px;}
    .t-month:first-child{margin-top:0;}
    .t-entry{padding:20px 0;border-bottom:1px solid #f0ebe4;cursor:pointer;transition:all 0.2s;}
    .t-entry:hover{padding-left:8px;}
    .t-entry h2{font-size:18px;font-weight:600;line-height:1.4;margin-bottom:4px;}
    .t-entry .excerpt{font-family:'Inter',sans-serif;font-size:13px;color:#a09486;line-height:1.7;}
    .t-entry .t-meta{display:flex;gap:16px;margin-top:8px;font-family:'Inter',sans-serif;font-size:11px;color:#c4b8aa;}
    .j-footer{max-width:720px;margin:0 auto;padding:48px 40px;font-family:'Inter',sans-serif;font-size:12px;color:#c4b8aa;text-align:center;border-top:1px solid #ede6dd;}
  `, `
    <nav class="j-nav">
      <div class="j-nav-brand">夏日手记</div>
      <div class="j-nav-links"><a href="#">归档</a><a href="#">标签</a><a href="#">关于我</a></div>
    </nav>
    <section class="j-hero">
      <h1>用文字留住<br><em>那些会消失的瞬间</em></h1>
      <p class="intro">关于代码的美学、阅读的余韵、以及生活中那些不值一提却值得记住的小事。</p>
    </section>
    <div class="j-divider"><hr></div>
    <section class="timeline">
      <div class="t-month">四月 2026</div>
      <div class="t-entry">
        <h2>厨房里的系统设计</h2>
        <p class="excerpt">今天做了一道复杂的菜，突然意识到做菜和系统设计惊人地相似：你需要预处理、并行执行、还有优雅地处理异常……</p>
        <div class="t-meta"><span>4月10日</span><span>·</span><span>生活 / 技术</span><span>·</span><span>5分钟</span></div>
      </div>
      <div class="t-entry">
        <h2>为什么好的 API 像好的散文</h2>
        <p class="excerpt">命名是最被低估的设计行为。一个好的函数名应该让调用者忘记它的实现，就像好的散文让你忘记文字本身。</p>
        <div class="t-meta"><span>4月2日</span><span>·</span><span>技术</span><span>·</span><span>8分钟</span></div>
      </div>
      <div class="t-month">三月 2026</div>
      <div class="t-entry">
        <h2>重新学会手写笔记</h2>
        <p class="excerpt">买了一支好钢笔之后，我开始重新手写。屏幕让思考变快，纸笔让思考变深。两者不矛盾。</p>
        <div class="t-meta"><span>3月28日</span><span>·</span><span>生活</span><span>·</span><span>4分钟</span></div>
      </div>
      <div class="t-entry">
        <h2>一杯咖啡的等待时间</h2>
        <p class="excerpt">手冲咖啡需要四分钟。我开始用这四分钟做正念练习——不看手机，只看水慢慢滤过咖啡粉。</p>
        <div class="t-meta"><span>3月15日</span><span>·</span><span>生活</span><span>·</span><span>3分钟</span></div>
      </div>
    </section>
    <footer class="j-footer">夏日手记 · 在这个匆忙的时代，慢慢写</footer>
  `),
},

// ──────────────────────── 3. 独立杂志 (Indie Mag) ────────────────────────
magazine: {
  name: '独立杂志',
  subtitle: '考究的排版、深度长文、期刊式结构',
  tags: ['editorial', 'longform', 'typography'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC','Playfair Display',Georgia,serif;background:#f5f0eb;color:#2c2420;min-height:100vh;}
    .m-header{text-align:center;padding:28px 40px;border-bottom:3px double #2c2420;}
    .m-header h1{font-family:'Playfair Display',serif;font-size:36px;font-weight:900;letter-spacing:-0.02em;}
    .m-header .tagline{font-family:'Inter',sans-serif;font-size:11px;color:#8a7e74;margin-top:4px;font-style:italic;}
    .m-dateline{text-align:center;padding:10px;font-family:'Inter',sans-serif;font-size:10px;color:#aaa;border-bottom:1px solid #ddd;letter-spacing:0.1em;text-transform:uppercase;}
    .m-featured{max-width:800px;margin:0 auto;padding:48px 40px;border-bottom:1px solid #d9d2c9;}
    .m-featured .category{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#b8a898;margin-bottom:12px;}
    .m-featured h2{font-size:32px;font-weight:900;line-height:1.25;letter-spacing:-0.01em;}
    .m-featured .lead{font-size:18px;line-height:1.8;color:#5a5048;margin-top:16px;font-style:italic;}
    .m-featured .byline{font-family:'Inter',sans-serif;font-size:11px;color:#aaa;margin-top:16px;}
    .m-grid{display:grid;grid-template-columns:1fr 1px 1fr;gap:0;max-width:800px;margin:0 auto;padding:32px 40px;}
    .m-col-div{background:#d9d2c9;}
    .m-col{padding:0 24px;}
    .m-col:first-child{padding-left:0;}
    .m-col:last-child{padding-right:0;}
    .m-article{padding:20px 0;border-bottom:1px solid #e8e2da;}
    .m-article:last-child{border-bottom:none;}
    .m-article .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#b8a898;margin-bottom:6px;}
    .m-article h3{font-size:17px;font-weight:700;line-height:1.35;margin-bottom:6px;}
    .m-article p{font-size:13px;line-height:1.7;color:#7a6e64;}
    .m-quote{max-width:800px;margin:0 auto;padding:36px 40px;border-top:1px solid #d9d2c9;border-bottom:1px solid #d9d2c9;}
    .m-quote blockquote{font-size:20px;line-height:1.6;font-style:italic;color:#5a5048;text-align:center;max-width:560px;margin:0 auto;}
    .m-quote cite{display:block;text-align:center;font-family:'Inter',sans-serif;font-size:11px;color:#aaa;margin-top:12px;font-style:normal;}
    .m-footer{text-align:center;padding:24px;border-top:3px double #2c2420;font-family:'Inter',sans-serif;font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:0.1em;max-width:800px;margin:0 auto;}
  `, `
    <header class="m-header">
      <h1>夏天的爱人</h1>
      <div class="tagline">一份关于代码、文字与生活的个人刊物</div>
    </header>
    <div class="m-dateline">第十二期 · 2026 年 4 月</div>
    <section class="m-featured">
      <div class="category">本期特稿</div>
      <h2>在代码与诗意之间，<br>寻找表达的可能</h2>
      <p class="lead">技术写作不应该只是教程。它可以是一种审美实践——关于如何用精确的语言，描述模糊的直觉。</p>
      <div class="byline">约 12 分钟阅读</div>
    </section>
    <div class="m-grid">
      <div class="m-col">
        <div class="m-article"><div class="cat">技术</div><h3>函数式编程的哲学根源</h3><p>从 Lambda 演算到日常编码——为什么不可变性不只是一个技术选择，而是一种世界观。</p></div>
        <div class="m-article"><div class="cat">随笔</div><h3>Rust 的第 100 天</h3><p>从"这个编译器是不是针对我"到"原来它在保护我"。关于信任一个严格朋友的故事。</p></div>
      </div>
      <div class="m-col-div"></div>
      <div class="m-col">
        <div class="m-article"><div class="cat">阅读</div><h3>重读《禅与摩托车维修艺术》</h3><p>Quality 不是一个形容词——它发生在你和世界之间的那个瞬间。时隔五年再读，有了完全不同的理解。</p></div>
        <div class="m-article"><div class="cat">生活</div><h3>极简生活的真实代价</h3><p>扔掉东西是容易的。困难的是面对扔掉之后的空白，和「为什么是这些」的追问。</p></div>
      </div>
    </div>
    <div class="m-quote">
      <blockquote>"完美不是无可添加，而是无可删减。"</blockquote>
      <cite>— Antoine de Saint-Exupéry</cite>
    </div>
    <footer class="m-footer">夏天的爱人 · 不定期出刊 · 2024 年创刊</footer>
  `),
},

// ──────────────────────── 4. 开发者笔记 (Dev Notes) ────────────────────────
devnotes: {
  name: '开发者笔记',
  subtitle: '代码友好、等宽字体、简洁直接',
  tags: ['dev', 'monospace', 'minimal'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#fdfdfc;color:#1a1a1a;min-height:100vh;}
    .d-nav{display:flex;justify-content:space-between;align-items:center;padding:20px 32px;max-width:760px;margin:0 auto;border-bottom:1px solid #eee;}
    .d-nav-brand{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:#1a1a1a;}
    .d-nav-brand .cursor{color:#6366f1;animation:blink 1s step-end infinite;}
    @keyframes blink{50%{opacity:0;}}
    .d-nav-links{display:flex;gap:20px;font-size:12px;color:#aaa;font-family:'JetBrains Mono',monospace;}
    .d-nav-links a:hover{color:#1a1a1a;}
    .d-hero{max-width:760px;margin:0 auto;padding:64px 32px 40px;}
    .d-hero p{font-size:15px;color:#666;line-height:1.8;max-width:520px;}
    .d-hero .greeting{font-family:'JetBrains Mono',monospace;font-size:13px;color:#6366f1;margin-bottom:12px;}
    .d-hero h1{font-size:28px;font-weight:800;letter-spacing:-0.03em;line-height:1.3;margin-bottom:12px;}
    .d-section{max-width:760px;margin:0 auto;padding:0 32px 40px;}
    .d-section-title{font-family:'JetBrains Mono',monospace;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:16px;display:flex;align-items:center;gap:8px;}
    .d-section-title::after{content:'';flex:1;height:1px;background:#eee;}
    .d-post{padding:16px 0;border-bottom:1px solid #f5f5f5;cursor:pointer;transition:all 0.15s;}
    .d-post:hover{padding-left:8px;}
    .d-post h3{font-size:15px;font-weight:600;margin-bottom:4px;}
    .d-post .desc{font-size:13px;color:#999;line-height:1.5;}
    .d-post .post-meta{display:flex;gap:12px;margin-top:6px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#ccc;}
    .d-post .post-meta .tag{color:#6366f1;background:#f0f0ff;padding:1px 6px;border-radius:3px;}
    .d-topics{max-width:760px;margin:0 auto;padding:0 32px 40px;display:flex;flex-wrap:wrap;gap:8px;}
    .d-topic{font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 12px;border:1px solid #eee;border-radius:6px;color:#888;cursor:pointer;transition:all 0.15s;}
    .d-topic:hover{border-color:#6366f1;color:#6366f1;}
    .d-topic .count{color:#ccc;margin-left:4px;}
    .d-footer{max-width:760px;margin:0 auto;padding:32px;border-top:1px solid #eee;font-family:'JetBrains Mono',monospace;font-size:11px;color:#ccc;display:flex;justify-content:space-between;}
  `, `
    <nav class="d-nav">
      <div class="d-nav-brand">xiatian<span class="cursor">_</span></div>
      <div class="d-nav-links"><a href="#">posts</a><a href="#">topics</a><a href="#">about</a><a href="#">rss</a></div>
    </nav>
    <section class="d-hero">
      <div class="greeting">// hello, world</div>
      <h1>写代码，也写关于代码的文字</h1>
      <p>全栈开发者。记录技术笔记、读书感悟和偶尔的生活碎片。所有文章都是写给半年后的自己看的。</p>
    </section>
    <div class="d-section">
      <div class="d-section-title">recent posts</div>
      <div class="d-post">
        <h3>系统设计的第一性原理</h3>
        <p class="desc">不从最佳实践出发，而是从约束条件推导。为什么同一个问题在不同上下文中有不同的"正确"答案。</p>
        <div class="post-meta"><span>2026-04-10</span><span class="tag">systems</span><span>8 min</span></div>
      </div>
      <div class="d-post">
        <h3>Rust 所有权模型的心智模型</h3>
        <p class="desc">不讲语法，只讲直觉。用三个类比帮你建立对 borrow checker 的正确预期。</p>
        <div class="post-meta"><span>2026-04-02</span><span class="tag">rust</span><span>12 min</span></div>
      </div>
      <div class="d-post">
        <h3>为什么我不用 ORM</h3>
        <p class="desc">ORM 解决了错误的问题。真正的痛点不是写 SQL，而是管理 schema 和迁移。</p>
        <div class="post-meta"><span>2026-03-22</span><span class="tag">database</span><span>6 min</span></div>
      </div>
      <div class="d-post">
        <h3>CSS 容器查询改变了一切</h3>
        <p class="desc">终于可以让组件自己决定长什么样了，不再依赖视口宽度。这才是真正的组件化。</p>
        <div class="post-meta"><span>2026-03-15</span><span class="tag">css</span><span>5 min</span></div>
      </div>
    </div>
    <div class="d-section">
      <div class="d-section-title">topics</div>
    </div>
    <div class="d-topics">
      <span class="d-topic">rust <span class="count">8</span></span>
      <span class="d-topic">systems <span class="count">6</span></span>
      <span class="d-topic">frontend <span class="count">11</span></span>
      <span class="d-topic">database <span class="count">4</span></span>
      <span class="d-topic">css <span class="count">5</span></span>
      <span class="d-topic">reading <span class="count">7</span></span>
      <span class="d-topic">life <span class="count">3</span></span>
    </div>
    <footer class="d-footer"><span>© 2026 xiatian</span><span>built with astro</span></footer>
  `),
},

// ═══════════════════ ROUND 2 VARIANTS ═══════════════════

// ── Garden → Notion-like vs Wiki-like ──
'garden-notion': {
  name: '花园 · Notion 风格',
  subtitle: '清爽界面、数据库视图、属性标签',
  tags: ['notion', 'database', 'structured'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#fff;color:#37352f;min-height:100vh;}
    .gn-sidebar{position:fixed;left:0;top:0;bottom:0;width:240px;background:#fbfbfa;border-right:1px solid #f0efec;padding:20px 12px;display:flex;flex-direction:column;}
    .gn-sidebar-brand{font-size:14px;font-weight:600;padding:8px 12px;margin-bottom:16px;}
    .gn-sidebar a{display:block;padding:6px 12px;border-radius:4px;font-size:13px;color:#787774;transition:background 0.1s;margin-bottom:1px;}
    .gn-sidebar a:hover{background:#f0efec;}
    .gn-sidebar a.active{background:#f0efec;color:#37352f;font-weight:500;}
    .gn-sidebar .section{font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#b4b4b0;padding:16px 12px 6px;font-weight:600;}
    .gn-main{margin-left:240px;padding:48px 60px;max-width:900px;}
    .gn-main h1{font-size:32px;font-weight:700;letter-spacing:-0.02em;margin-bottom:4px;}
    .gn-main .subtitle{font-size:14px;color:#999;margin-bottom:32px;}
    .gn-views{display:flex;gap:0;border-bottom:1px solid #eee;margin-bottom:20px;}
    .gn-views button{background:none;border:none;padding:8px 16px;font-size:13px;color:#aaa;cursor:pointer;border-bottom:2px solid transparent;font-family:inherit;}
    .gn-views button.active{color:#37352f;border-bottom-color:#37352f;}
    .gn-table{width:100%;border-collapse:collapse;}
    .gn-table th{text-align:left;padding:8px 12px;font-size:11px;color:#999;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid #eee;background:#fafafa;}
    .gn-table td{padding:10px 12px;border-bottom:1px solid #f5f5f5;font-size:13px;cursor:pointer;}
    .gn-table tr:hover td{background:#fafafa;}
    .gn-table .title-cell{font-weight:500;color:#37352f;}
    .gn-table .status{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:500;}
    .gn-table .status.done{background:#dbeddb;color:#2d6a2e;}
    .gn-table .status.wip{background:#fdecc8;color:#9a6700;}
    .gn-table .status.seed{background:#d3e5ef;color:#2e6b8a;}
    .gn-table .tag-cell{font-size:11px;color:#aaa;}
  `, `
    <aside class="gn-sidebar">
      <div class="gn-sidebar-brand">🌱 夏天的花园</div>
      <a class="active" href="#">全部笔记</a>
      <a href="#">最近更新</a>
      <a href="#">随机漫步</a>
      <div class="section">主题</div>
      <a href="#">系统设计</a>
      <a href="#">Rust</a>
      <a href="#">前端</a>
      <a href="#">阅读笔记</a>
      <a href="#">生活随想</a>
      <div class="section">其他</div>
      <a href="#">知识图谱</a>
      <a href="#">关于花园</a>
      <a href="#">关于我</a>
    </aside>
    <main class="gn-main">
      <h1>全部笔记</h1>
      <p class="subtitle">42 篇笔记 · 按最近更新排序</p>
      <div class="gn-views"><button class="active">表格</button><button>卡片</button><button>时间线</button></div>
      <table class="gn-table">
        <thead><tr><th>标题</th><th>状态</th><th>主题</th><th>更新日期</th></tr></thead>
        <tbody>
          <tr><td class="title-cell">系统设计的第一性原理</td><td><span class="status done">常青</span></td><td class="tag-cell">系统设计</td><td class="tag-cell">Apr 10</td></tr>
          <tr><td class="title-cell">Rust 所有权与人生责任</td><td><span class="status wip">成长中</span></td><td class="tag-cell">Rust</td><td class="tag-cell">Apr 8</td></tr>
          <tr><td class="title-cell">为什么好的 API 像好的散文</td><td><span class="status done">常青</span></td><td class="tag-cell">设计</td><td class="tag-cell">Apr 2</td></tr>
          <tr><td class="title-cell">厨房里的系统思维</td><td><span class="status seed">萌芽</span></td><td class="tag-cell">生活</td><td class="tag-cell">Mar 28</td></tr>
          <tr><td class="title-cell">CSS 容器查询改变了一切</td><td><span class="status wip">成长中</span></td><td class="tag-cell">前端</td><td class="tag-cell">Mar 22</td></tr>
          <tr><td class="title-cell">重读《禅与摩托车维修艺术》</td><td><span class="status seed">萌芽</span></td><td class="tag-cell">阅读</td><td class="tag-cell">Mar 15</td></tr>
          <tr><td class="title-cell">静态网站的文艺复兴</td><td><span class="status wip">成长中</span></td><td class="tag-cell">前端</td><td class="tag-cell">Mar 10</td></tr>
        </tbody>
      </table>
    </main>
  `),
},

'garden-wiki': {
  name: '花园 · Wiki 风格',
  subtitle: '双栏链接、反向引用、内容为核心',
  tags: ['wiki', 'links', 'interconnected'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#f7f6f3;color:#37352f;min-height:100vh;}
    .gw-nav{display:flex;justify-content:space-between;align-items:center;padding:16px 32px;max-width:860px;margin:0 auto;font-size:13px;}
    .gw-nav-brand{font-weight:700;}
    .gw-nav-links{display:flex;gap:16px;color:#999;font-size:12px;}
    .gw-content{max-width:860px;margin:0 auto;padding:40px 32px;display:grid;grid-template-columns:1fr 260px;gap:48px;}
    .gw-article h1{font-size:28px;font-weight:800;letter-spacing:-0.02em;margin-bottom:8px;}
    .gw-article .meta{font-size:12px;color:#b4b4b0;margin-bottom:24px;display:flex;gap:12px;}
    .gw-article .meta .status{padding:2px 8px;border-radius:10px;background:#e8f5e9;color:#4caf50;font-size:11px;}
    .gw-article p{font-size:15px;line-height:1.85;color:#555;margin-bottom:16px;}
    .gw-article .wikilink{color:#6366f1;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px;cursor:pointer;}
    .gw-article .wikilink:hover{text-decoration-style:solid;}
    .gw-article h2{font-size:18px;font-weight:700;margin:28px 0 12px;letter-spacing:-0.01em;}
    .gw-aside{position:sticky;top:32px;align-self:start;}
    .gw-aside-section{margin-bottom:28px;}
    .gw-aside h4{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#b4b4b0;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #e8e8e4;}
    .gw-aside a{display:block;font-size:12px;color:#787774;padding:4px 0;cursor:pointer;transition:color 0.1s;}
    .gw-aside a:hover{color:#6366f1;}
    .gw-aside .toc a{padding-left:0;}
    .gw-aside .toc a.sub{padding-left:12px;color:#aaa;}
    .gw-footer{max-width:860px;margin:0 auto;padding:24px 32px;border-top:1px solid #e8e8e4;font-size:11px;color:#b4b4b0;}
  `, `
    <nav class="gw-nav">
      <div class="gw-nav-brand">🌱 夏天的花园</div>
      <div class="gw-nav-links"><a href="#">首页</a><a href="#">图谱</a><a href="#">随机</a><a href="#">关于</a></div>
    </nav>
    <div class="gw-content">
      <article class="gw-article">
        <h1>系统设计的第一性原理</h1>
        <div class="meta"><span class="status">🌳 常青</span><span>最后更新 Apr 10</span><span>·</span><span>被引用 5 次</span></div>
        <p>多数系统设计教程的问题在于：它们从"最佳实践"出发，而不是从<span class="wikilink">约束条件</span>出发。最佳实践是别人的答案，而你需要的是推导出自己的答案的方法。</p>
        <h2>从约束开始</h2>
        <p>任何系统设计的第一步不是选技术栈，而是明确约束。约束分三层：<span class="wikilink">业务约束</span>（必须满足）、技术约束（当前必须面对）和团队约束（经常被忽略）。</p>
        <p>这和<span class="wikilink">厨房里的系统思维</span>异曲同工：你不会在没有看冰箱之前就决定今晚做什么菜。约束决定方案，而不是反过来。</p>
        <h2>复杂度的代价</h2>
        <p>每一个架构决策都是在<span class="wikilink">简单性</span>和某种能力之间做交换。关键不是避免复杂度，而是确保你在<span class="wikilink">正确的地方</span>引入了复杂度。这需要你理解<span class="wikilink">Rust 所有权模型</span>教会我们的道理：每个选择都有成本。</p>
      </article>
      <aside class="gw-aside">
        <div class="gw-aside-section">
          <h4>目录</h4>
          <div class="toc">
            <a href="#">从约束开始</a>
            <a href="#">复杂度的代价</a>
          </div>
        </div>
        <div class="gw-aside-section">
          <h4>反向链接</h4>
          <a href="#">→ 厨房里的系统思维</a>
          <a href="#">→ Rust 所有权与人生责任</a>
          <a href="#">→ 为什么好的 API 像好的散文</a>
          <a href="#">→ 静态网站的文艺复兴</a>
          <a href="#">→ 约束即自由</a>
        </div>
        <div class="gw-aside-section">
          <h4>相关主题</h4>
          <a href="#">系统设计</a>
          <a href="#">架构</a>
          <a href="#">第一性原理</a>
        </div>
      </aside>
    </div>
    <footer class="gw-footer">🌱 夏天的花园 · 42 篇笔记 · 128 条链接</footer>
  `),
},

// ── Journal → 暖色有图 vs 纯文字 ──
'journal-rich': {
  name: '手帐 · 图文并茂',
  subtitle: '配图、手绘元素、更像实体杂志',
  tags: ['visual', 'rich', 'illustrated'],
  html: () => wrapPage(`
    body{font-family:'Lora','Noto Serif SC',Georgia,serif;background:#faf7f2;color:#3d3226;min-height:100vh;}
    .jr-nav{display:flex;justify-content:space-between;align-items:center;padding:20px 40px;max-width:720px;margin:0 auto;}
    .jr-brand{font-family:'DM Serif Display',serif;font-size:18px;}
    .jr-links{display:flex;gap:16px;font-family:'Inter',sans-serif;font-size:12px;color:#b8a898;}
    .jr-hero{max-width:720px;margin:0 auto;padding:48px 40px 32px;display:grid;grid-template-columns:1fr 200px;gap:32px;align-items:center;}
    .jr-hero h1{font-family:'DM Serif Display',serif;font-size:30px;font-weight:400;line-height:1.35;}
    .jr-hero h1 em{color:#c4956a;font-style:italic;}
    .jr-hero p{font-family:'Inter',sans-serif;font-size:13px;color:#a09486;margin-top:10px;line-height:1.7;}
    .jr-hero-img{aspect-ratio:1;border-radius:50%;background:linear-gradient(135deg,#e8ddd0,#d4c4ae);display:flex;align-items:center;justify-content:center;font-size:64px;}
    .jr-featured{max-width:720px;margin:0 auto;padding:0 40px 32px;}
    .jr-featured-card{border-radius:16px;overflow:hidden;background:#fff;border:1px solid #ede6dd;cursor:pointer;transition:all 0.2s;}
    .jr-featured-card:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,0.05);}
    .jr-featured-img{height:200px;background:linear-gradient(135deg,#e8ddd0,#c4956a);display:flex;align-items:flex-end;padding:20px;}
    .jr-featured-img span{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.7);background:rgba(0,0,0,0.2);padding:4px 10px;border-radius:10px;}
    .jr-featured-body{padding:24px;}
    .jr-featured-body h2{font-size:20px;margin-bottom:6px;}
    .jr-featured-body p{font-family:'Inter',sans-serif;font-size:13px;color:#a09486;line-height:1.7;}
    .jr-list{max-width:720px;margin:0 auto;padding:0 40px 40px;}
    .jr-item{display:grid;grid-template-columns:100px 1fr;gap:16px;padding:16px 0;border-bottom:1px solid #f0ebe4;cursor:pointer;transition:all 0.15s;}
    .jr-item:hover{padding-left:4px;}
    .jr-item-img{aspect-ratio:1;border-radius:10px;background:linear-gradient(135deg,#e0d5c5,#c8bba8);}
    .jr-item-text h3{font-size:16px;font-weight:600;line-height:1.3;margin-bottom:4px;}
    .jr-item-text p{font-family:'Inter',sans-serif;font-size:12px;color:#a09486;line-height:1.5;}
    .jr-item-text .meta{font-family:'Inter',sans-serif;font-size:11px;color:#c4b8aa;margin-top:6px;}
    .jr-footer{max-width:720px;margin:0 auto;padding:32px 40px;text-align:center;font-family:'Inter',sans-serif;font-size:11px;color:#c4b8aa;border-top:1px solid #ede6dd;}
  `, `
    <nav class="jr-nav"><div class="jr-brand">夏日手记</div><div class="jr-links"><a href="#">归档</a><a href="#">影集</a><a href="#">关于</a></div></nav>
    <section class="jr-hero">
      <div><h1>用文字和影像<br>留住<em>消失的瞬间</em></h1><p>一个人的生活杂志。关于代码、阅读、做菜和窗外的风景。</p></div>
      <div class="jr-hero-img">🌿</div>
    </section>
    <div class="jr-featured">
      <div class="jr-featured-card">
        <div class="jr-featured-img"><span>本周推荐</span></div>
        <div class="jr-featured-body"><h2>厨房里的系统设计</h2><p>今天做了一道复杂的菜，突然意识到做菜和系统设计惊人地相似。</p></div>
      </div>
    </div>
    <div class="jr-list">
      <div class="jr-item"><div class="jr-item-img"></div><div class="jr-item-text"><h3>为什么好的 API 像好的散文</h3><p>命名是最被低估的设计行为。</p><div class="meta">4月2日 · 8分钟</div></div></div>
      <div class="jr-item"><div class="jr-item-img"></div><div class="jr-item-text"><h3>重新学会手写笔记</h3><p>屏幕让思考变快，纸笔让思考变深。</p><div class="meta">3月28日 · 4分钟</div></div></div>
      <div class="jr-item"><div class="jr-item-img"></div><div class="jr-item-text"><h3>一杯咖啡的等待时间</h3><p>手冲需要四分钟。我用这四分钟做正念练习。</p><div class="meta">3月15日 · 3分钟</div></div></div>
    </div>
    <footer class="jr-footer">夏日手记 · 2024 年至今 · 偶尔更新</footer>
  `),
},

'journal-pure': {
  name: '手帐 · 纯文字',
  subtitle: '没有装饰、只有文字和呼吸',
  tags: ['text-only', 'intimate', 'zen'],
  html: () => wrapPage(`
    body{font-family:'Lora','Noto Serif SC',Georgia,serif;background:#faf7f2;color:#3d3226;min-height:100vh;display:flex;flex-direction:column;}
    .jp-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 40px;max-width:600px;margin:0 auto;width:100%;}
    .jp-brand{font-family:'DM Serif Display',serif;font-size:18px;}
    .jp-links{font-family:'Inter',sans-serif;font-size:12px;color:#b8a898;display:flex;gap:16px;}
    .jp-main{flex:1;max-width:600px;margin:0 auto;padding:60px 40px;width:100%;}
    .jp-greeting{font-family:'DM Serif Display',serif;font-size:28px;font-weight:400;line-height:1.45;margin-bottom:8px;}
    .jp-greeting em{color:#c4956a;font-style:italic;}
    .jp-sub{font-family:'Inter',sans-serif;font-size:13px;color:#a09486;line-height:1.8;margin-bottom:48px;}
    .jp-sep{width:32px;height:1px;background:#d8cec2;margin-bottom:40px;}
    .jp-entry{margin-bottom:36px;cursor:pointer;transition:all 0.2s;}
    .jp-entry:hover{transform:translateX(4px);}
    .jp-entry .date{font-family:'Inter',sans-serif;font-size:11px;color:#c4b8aa;margin-bottom:6px;letter-spacing:0.02em;}
    .jp-entry h2{font-size:19px;font-weight:600;line-height:1.4;margin-bottom:6px;}
    .jp-entry p{font-family:'Inter',sans-serif;font-size:13px;color:#a09486;line-height:1.7;}
    .jp-footer{max-width:600px;margin:0 auto;padding:32px 40px;font-family:'Inter',sans-serif;font-size:11px;color:#c4b8aa;width:100%;}
  `, `
    <nav class="jp-nav"><div class="jp-brand">夏日手记</div><div class="jp-links"><a href="#">全部</a><a href="#">关于</a></div></nav>
    <main class="jp-main">
      <div class="jp-greeting">你好，<br>欢迎来到我的<em>小世界</em></div>
      <p class="jp-sub">这里没有什么宏大叙事。只有日常的碎片——写代码的间隙、读书的随想、和窗外不断变化的光线。</p>
      <div class="jp-sep"></div>
      <div class="jp-entry"><div class="date">2026 年 4 月 10 日</div><h2>厨房里的系统设计</h2><p>今天做了一道复杂的菜。忽然意识到做菜和系统设计惊人地相似。</p></div>
      <div class="jp-entry"><div class="date">2026 年 4 月 2 日</div><h2>为什么好的 API 像好的散文</h2><p>命名是最被低估的设计行为。一个好的函数名应该让读者忘记实现。</p></div>
      <div class="jp-entry"><div class="date">2026 年 3 月 28 日</div><h2>重新学会手写笔记</h2><p>买了一支好钢笔之后，我开始重新手写。屏幕让思考变快，纸笔让思考变深。</p></div>
      <div class="jp-entry"><div class="date">2026 年 3 月 15 日</div><h2>一杯咖啡的等待时间</h2><p>手冲需要四分钟。我开始用这四分钟做正念练习。</p></div>
    </main>
    <footer class="jp-footer">慢慢写，慢慢活。</footer>
  `),
},

// ── Magazine → 深色 vs 浅色 ──
'magazine-dark': {
  name: '杂志 · 深色沉浸',
  subtitle: '深色背景、白色文字、夜间阅读感',
  tags: ['dark', 'immersive', 'reading'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC','Lora',Georgia,serif;background:#141210;color:#d4cec6;min-height:100vh;}
    .md-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 40px;max-width:800px;margin:0 auto;}
    .md-brand{font-family:'Playfair Display',serif;font-size:22px;color:#e8e2da;font-weight:700;}
    .md-links{display:flex;gap:20px;font-family:'Inter',sans-serif;font-size:12px;color:#666;}
    .md-featured{max-width:800px;margin:0 auto;padding:60px 40px 40px;border-bottom:1px solid #2a2622;}
    .md-featured .cat{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#666;margin-bottom:12px;}
    .md-featured h1{font-family:'Playfair Display',serif;font-size:36px;font-weight:700;line-height:1.25;color:#e8e2da;letter-spacing:-0.01em;}
    .md-featured .lead{font-size:17px;line-height:1.8;color:#8a8078;margin-top:16px;font-style:italic;}
    .md-featured .byline{font-family:'Inter',sans-serif;font-size:11px;color:#555;margin-top:16px;}
    .md-articles{max-width:800px;margin:0 auto;padding:32px 40px;}
    .md-art{padding:24px 0;border-bottom:1px solid #1e1c1a;cursor:pointer;transition:all 0.2s;}
    .md-art:hover{padding-left:8px;}
    .md-art .cat{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:#555;margin-bottom:6px;}
    .md-art h2{font-family:'Playfair Display',serif;font-size:20px;color:#d4cec6;font-weight:600;line-height:1.3;margin-bottom:6px;}
    .md-art p{font-size:14px;color:#666;line-height:1.6;}
    .md-quote{max-width:800px;margin:0 auto;padding:48px 40px;text-align:center;border-top:1px solid #2a2622;border-bottom:1px solid #2a2622;}
    .md-quote blockquote{font-family:'Playfair Display',serif;font-size:22px;font-style:italic;color:#8a8078;line-height:1.5;max-width:500px;margin:0 auto;}
    .md-quote cite{display:block;font-family:'Inter',sans-serif;font-size:11px;color:#444;margin-top:12px;font-style:normal;}
    .md-footer{max-width:800px;margin:0 auto;padding:32px 40px;font-family:'Inter',sans-serif;font-size:10px;color:#444;text-align:center;text-transform:uppercase;letter-spacing:0.1em;}
  `, `
    <nav class="md-nav"><div class="md-brand">夏天的爱人</div><div class="md-links"><a href="#">文章</a><a href="#">归档</a><a href="#">关于</a></div></nav>
    <section class="md-featured">
      <div class="cat">本期特稿</div>
      <h1>在代码与诗意之间，<br>寻找表达的可能</h1>
      <p class="lead">技术写作不应该只是教程。它可以是一种审美实践——关于如何用精确的语言，描述模糊的直觉。</p>
      <div class="byline">约 12 分钟阅读</div>
    </section>
    <div class="md-articles">
      <div class="md-art"><div class="cat">技术</div><h2>函数式编程的哲学根源</h2><p>从 Lambda 演算到日常编码——不可变性不只是技术选择，而是一种世界观。</p></div>
      <div class="md-art"><div class="cat">随笔</div><h2>Rust 的第 100 天</h2><p>关于信任一个严格但正确的朋友的故事。</p></div>
      <div class="md-art"><div class="cat">阅读</div><h2>重读《禅与摩托车维修艺术》</h2><p>Quality 发生在你和世界之间的那个瞬间。</p></div>
      <div class="md-art"><div class="cat">生活</div><h2>极简生活的真实代价</h2><p>扔掉东西是容易的。困难的是面对之后的空白。</p></div>
    </div>
    <div class="md-quote"><blockquote>"我们写的代码，其实是写给未来的信。"</blockquote><cite>— 夏天的爱人 · 第八期</cite></div>
    <footer class="md-footer">不定期出刊 · 2024 年创刊</footer>
  `),
},

'magazine-light': {
  name: '杂志 · 经典浅色',
  subtitle: '纸张质感、多栏报纸排版、严肃阅读',
  tags: ['light', 'newspaper', 'classic'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f8f5f0;color:#222;min-height:100vh;}
    .ml-header{text-align:center;padding:28px 40px;border-bottom:3px double #222;max-width:860px;margin:0 auto;}
    .ml-header h1{font-family:'Playfair Display',serif;font-size:38px;font-weight:900;}
    .ml-header .tagline{font-size:12px;color:#888;font-style:italic;margin-top:2px;}
    .ml-dateline{text-align:center;padding:10px;font-family:'Inter',sans-serif;font-size:10px;color:#999;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #ddd;max-width:860px;margin:0 auto;}
    .ml-featured{max-width:860px;margin:0 auto;padding:40px 40px 32px;border-bottom:1px solid #d9d2c9;}
    .ml-featured .cat{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;margin-bottom:10px;}
    .ml-featured h2{font-size:30px;font-weight:900;line-height:1.25;}
    .ml-featured .lead{font-size:17px;line-height:1.8;color:#555;margin-top:12px;font-style:italic;max-width:600px;}
    .ml-grid{display:grid;grid-template-columns:1fr 1px 1fr 1px 1fr;gap:0;max-width:860px;margin:0 auto;padding:28px 40px;}
    .ml-div{background:#d9d2c9;}
    .ml-col{padding:0 20px;}
    .ml-col:first-child{padding-left:0;}
    .ml-col:last-child{padding-right:0;}
    .ml-col h3{font-size:16px;font-weight:700;line-height:1.35;margin-bottom:8px;}
    .ml-col .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#aaa;margin-bottom:6px;}
    .ml-col p{font-size:13px;line-height:1.7;color:#666;text-align:justify;}
    .ml-footer{text-align:center;padding:20px;border-top:3px double #222;font-family:'Inter',sans-serif;font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:0.1em;max-width:860px;margin:0 auto;}
  `, `
    <header class="ml-header"><h1>夏天的爱人</h1><div class="tagline">"在字里行间，找到回家的路"</div></header>
    <div class="ml-dateline">第十二期 · 2026 年 4 月 14 日 · 星期二</div>
    <section class="ml-featured">
      <div class="cat">本期特稿</div>
      <h2>当我们谈论架构时<br>我们谈论什么</h2>
      <p class="lead">软件架构不是一个决定，而是一系列决定的叠加。每一次 trade-off 都在塑造系统的性格。就像建筑一样，好的架构让人忘记它的存在。</p>
    </section>
    <div class="ml-grid">
      <div class="ml-col"><div class="cat">随笔</div><h3>极简生活的代价</h3><p>扔掉东西是容易的。困难的是面对扔掉之后的空白。极简主义的真正挑战不在于"少"，而在于"为什么是这些"。</p></div>
      <div class="ml-div"></div>
      <div class="ml-col"><div class="cat">技术</div><h3>Rust 的第 100 天</h3><p>从"编译器是不是针对我"到"原来它在保护我"。学习 Rust 的过程，像极了学习信任一个严格但正确的朋友。</p></div>
      <div class="ml-div"></div>
      <div class="ml-col"><div class="cat">阅读</div><h3>重读《禅与摩托车<br>维修艺术》</h3><p>Quality 不是一个形容词。它发生在你和世界之间的那个瞬间。时隔五年再读，理解完全不同。</p></div>
    </div>
    <footer class="ml-footer">不定期出刊 · 2024 年创刊</footer>
  `),
},

// ══════════════ ROUND 3: magazine-light refinements ══════════════

// ── magazine-light → 排版气质 ──
'ml-newspaper': {
  name: '报纸厚重感',
  subtitle: '双线边框、黑白强对比、密集栏目、masthead 大标题',
  tags: ['newspaper', 'dense', 'bold'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f4f0e8;color:#1a1a1a;min-height:100vh;}
    .np-header{text-align:center;padding:20px 40px 16px;border-bottom:4px solid #1a1a1a;max-width:900px;margin:0 auto;}
    .np-header .overline{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.2em;color:#999;margin-bottom:4px;}
    .np-header h1{font-family:'Playfair Display',serif;font-size:48px;font-weight:900;letter-spacing:-0.03em;line-height:1;}
    .np-header .underline{font-size:11px;color:#888;font-style:italic;margin-top:4px;}
    .np-datebar{display:flex;justify-content:space-between;max-width:900px;margin:0 auto;padding:8px 40px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#999;}
    .np-featured{max-width:900px;margin:0 auto;padding:28px 40px;border-bottom:1px solid #c8c0b4;display:grid;grid-template-columns:1fr 1fr;gap:32px;align-items:start;}
    .np-featured .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.15em;color:#aaa;margin-bottom:8px;}
    .np-featured h2{font-size:28px;font-weight:900;line-height:1.2;letter-spacing:-0.01em;margin-bottom:10px;}
    .np-featured .lead{font-size:15px;line-height:1.8;color:#555;font-style:italic;}
    .np-featured .right-col{border-left:1px solid #c8c0b4;padding-left:32px;}
    .np-featured .right-col h3{font-size:18px;font-weight:700;line-height:1.3;margin-bottom:6px;}
    .np-featured .right-col p{font-size:13px;line-height:1.7;color:#666;}
    .np-featured .right-col .divider{height:1px;background:#ddd5ca;margin:16px 0;}
    .np-cols{display:grid;grid-template-columns:1fr 1px 1fr 1px 1fr;gap:0;max-width:900px;margin:0 auto;padding:24px 40px;}
    .np-div{background:#c8c0b4;}
    .np-col{padding:0 18px;}
    .np-col:first-child{padding-left:0;}
    .np-col:last-child{padding-right:0;}
    .np-col .cat{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;margin-bottom:4px;}
    .np-col h4{font-size:14px;font-weight:700;line-height:1.35;margin-bottom:6px;}
    .np-col p{font-size:12px;line-height:1.65;color:#777;text-align:justify;}
    .np-col .item{padding:12px 0;border-bottom:1px solid #e4ddd4;}
    .np-col .item:last-child{border-bottom:none;}
    .np-footer{max-width:900px;margin:0 auto;padding:16px 40px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="np-header">
      <div class="overline">A Personal Publication on Code, Words & Life</div>
      <h1>夏天的爱人</h1>
      <div class="underline">"在字里行间，找到回家的路"</div>
    </header>
    <div class="np-datebar"><span>第十二期</span><span>2026 年 4 月 14 日 · 星期二</span><span>不定期出刊</span></div>
    <section class="np-featured">
      <div>
        <div class="cat">本期特稿</div>
        <h2>当我们谈论架构时<br>我们谈论什么</h2>
        <p class="lead">软件架构不是一个决定，而是一系列决定的叠加。每一次 trade-off 都在塑造系统的性格。就像建筑一样，好的架构让人忘记它的存在。</p>
      </div>
      <div class="right-col">
        <div class="cat">技术</div>
        <h3>函数式编程的哲学根源</h3>
        <p>从 Lambda 演算到日常编码——不可变性不只是技术选择，而是一种世界观。</p>
        <div class="divider"></div>
        <div class="cat">随笔</div>
        <h3>Rust 的第 100 天</h3>
        <p>从"编译器是不是针对我"到"原来它在保护我"。</p>
      </div>
    </section>
    <div class="np-cols">
      <div class="np-col">
        <div class="item"><div class="cat">阅读</div><h4>重读《禅与摩托车维修艺术》</h4><p>Quality 不是一个形容词。它发生在你和世界之间的那个瞬间。</p></div>
        <div class="item"><div class="cat">生活</div><h4>极简生活的真实代价</h4><p>扔掉东西是容易的。困难的是面对空白。</p></div>
      </div>
      <div class="np-div"></div>
      <div class="np-col">
        <div class="item"><div class="cat">技术</div><h4>为什么我不用 ORM</h4><p>真正的痛点不是写 SQL，而是管理 schema 迁移。</p></div>
        <div class="item"><div class="cat">设计</div><h4>CSS 容器查询改变了一切</h4><p>组件可以自己决定长什么样了。真正的组件化。</p></div>
      </div>
      <div class="np-div"></div>
      <div class="np-col">
        <div class="item"><div class="cat">生活</div><h4>厨房里的系统思维</h4><p>做菜和写代码的共同点出奇地多。</p></div>
        <div class="item"><div class="cat">阅读</div><h4>一杯咖啡的等待时间</h4><p>四分钟的正念练习从手冲开始。</p></div>
      </div>
    </div>
    <footer class="np-footer"><span>© 2026 夏天的爱人</span><span>2024 年创刊</span></footer>
  `),
},

'ml-literary': {
  name: '文艺留白感',
  subtitle: '大量留白、主次分明、衬线优雅、单栏为主',
  tags: ['literary', 'spacious', 'elegant'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC','Lora',Georgia,serif;background:#f6f3ee;color:#2c2420;min-height:100vh;}
    .lt-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 48px;max-width:740px;margin:0 auto;}
    .lt-brand{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;letter-spacing:-0.02em;}
    .lt-links{display:flex;gap:24px;font-family:'Inter',sans-serif;font-size:12px;color:#aaa;}
    .lt-hero{max-width:740px;margin:0 auto;padding:80px 48px 48px;}
    .lt-hero .issue{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#bbb;margin-bottom:20px;}
    .lt-hero h1{font-family:'Playfair Display',serif;font-size:38px;font-weight:700;line-height:1.3;letter-spacing:-0.01em;}
    .lt-hero .lead{font-size:18px;line-height:1.85;color:#7a6e64;margin-top:20px;font-style:italic;max-width:540px;}
    .lt-hero .read-link{display:inline-block;margin-top:20px;font-family:'Inter',sans-serif;font-size:12px;color:#2c2420;border-bottom:1px solid #2c2420;padding-bottom:2px;letter-spacing:0.02em;}
    .lt-sep{max-width:740px;margin:0 auto;padding:0 48px;}
    .lt-sep hr{border:none;height:1px;background:#ddd5ca;}
    .lt-list{max-width:740px;margin:0 auto;padding:36px 48px;}
    .lt-label{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#c4b8aa;margin-bottom:20px;}
    .lt-item{padding:20px 0;border-bottom:1px solid #e8e2da;cursor:pointer;transition:all 0.2s;}
    .lt-item:hover{padding-left:8px;}
    .lt-item .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#c4b8aa;margin-bottom:4px;}
    .lt-item h3{font-size:19px;font-weight:600;line-height:1.35;margin-bottom:4px;}
    .lt-item p{font-size:14px;color:#8a7e74;line-height:1.6;}
    .lt-footer{max-width:740px;margin:0 auto;padding:40px 48px;border-top:1px solid #ddd5ca;text-align:center;font-family:'Inter',sans-serif;font-size:10px;color:#c4b8aa;letter-spacing:0.05em;}
  `, `
    <nav class="lt-nav"><div class="lt-brand">夏天的爱人</div><div class="lt-links"><a href="#">文章</a><a href="#">归档</a><a href="#">关于</a></div></nav>
    <section class="lt-hero">
      <div class="issue">第十二期 · 2026 年 4 月</div>
      <h1>在代码与诗意之间，<br>寻找表达的可能</h1>
      <p class="lead">技术写作不应该只是教程。它可以是一种审美实践——关于如何用精确的语言，描述模糊的直觉。</p>
      <a class="read-link" href="#">阅读全文 →</a>
    </section>
    <div class="lt-sep"><hr></div>
    <section class="lt-list">
      <div class="lt-label">更多文章</div>
      <div class="lt-item"><div class="cat">技术</div><h3>当我们谈论架构时我们谈论什么</h3><p>软件架构不是一个决定，而是一系列决定的叠加。</p></div>
      <div class="lt-item"><div class="cat">随笔</div><h3>Rust 的第 100 天</h3><p>关于信任一个严格但正确的朋友的故事。</p></div>
      <div class="lt-item"><div class="cat">阅读</div><h3>重读《禅与摩托车维修艺术》</h3><p>Quality 发生在你和世界之间的那个瞬间。</p></div>
      <div class="lt-item"><div class="cat">生活</div><h3>极简生活的真实代价</h3><p>扔掉东西是容易的。困难的是面对之后的空白。</p></div>
    </section>
    <footer class="lt-footer">夏天的爱人 · 一份关于代码与生活的个人刊物 · 2024 年创刊</footer>
  `),
},

// ══════════════ ROUND 4: newspaper refinements ══════════════

'ml-np-dense': {
  name: '报纸 · 密集三栏',
  subtitle: '最大化信息密度，同时展示尽可能多的内容',
  tags: ['dense', 'three-column', 'information'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f2eee5;color:#1a1a1a;min-height:100vh;}
    .hd{text-align:center;padding:16px 36px 12px;border-bottom:4px solid #1a1a1a;max-width:940px;margin:0 auto;}
    .hd .over{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.2em;color:#aaa;}
    .hd h1{font-family:'Playfair Display',serif;font-size:44px;font-weight:900;letter-spacing:-0.03em;line-height:1;}
    .db{display:flex;justify-content:space-between;max-width:940px;margin:0 auto;padding:6px 36px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;}
    .ft{max-width:940px;margin:0 auto;padding:24px 36px;border-bottom:2px solid #c4baa8;text-align:center;}
    .ft .cat{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.15em;color:#aaa;margin-bottom:8px;}
    .ft h2{font-size:32px;font-weight:900;line-height:1.15;letter-spacing:-0.02em;max-width:600px;margin:0 auto 8px;}
    .ft .ld{font-size:15px;line-height:1.7;color:#666;font-style:italic;max-width:520px;margin:0 auto;}
    .cols{display:grid;grid-template-columns:1fr 1px 1fr 1px 1fr;gap:0;max-width:940px;margin:0 auto;padding:20px 36px;}
    .dv{background:#c4baa8;}
    .cl{padding:0 16px;}
    .cl:first-child{padding-left:0;}
    .cl:last-child{padding-right:0;}
    .it{padding:10px 0;border-bottom:1px solid #e2dbd0;}
    .it:last-child{border-bottom:none;}
    .it .c{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:3px;}
    .it h4{font-size:13px;font-weight:700;line-height:1.3;margin-bottom:4px;}
    .it p{font-size:11px;line-height:1.6;color:#888;text-align:justify;}
    .bot{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;max-width:940px;margin:0 auto;padding:16px 36px;border-top:1px solid #c4baa8;}
    .bot-it{padding:10px 0;}
    .bot-it .c{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:2px;}
    .bot-it h5{font-size:12px;font-weight:700;line-height:1.3;}
    .ftr{max-width:940px;margin:0 auto;padding:12px 36px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:8px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="hd"><div class="over">A Personal Publication</div><h1>夏天的爱人</h1></header>
    <div class="db"><span>第十二期</span><span>2026 年 4 月 14 日 星期二</span><span>不定期出刊</span></div>
    <section class="ft">
      <div class="cat">本期特稿</div>
      <h2>当我们谈论架构时<br>我们谈论什么</h2>
      <p class="ld">软件架构不是一个决定，而是一系列决定的叠加。每一次 trade-off 都在塑造系统的性格。</p>
    </section>
    <div class="cols">
      <div class="cl">
        <div class="it"><div class="c">技术</div><h4>函数式编程的哲学根源</h4><p>从 Lambda 演算到日常编码。不可变性不只是技术选择。</p></div>
        <div class="it"><div class="c">随笔</div><h4>Rust 的第 100 天</h4><p>从敌意到信任。编译器不是针对你，它在保护你。</p></div>
        <div class="it"><div class="c">技术</div><h4>为什么我不用 ORM</h4><p>痛点不是写 SQL，而是管理 schema 迁移。</p></div>
      </div>
      <div class="dv"></div>
      <div class="cl">
        <div class="it"><div class="c">阅读</div><h4>重读《禅与摩托车维修艺术》</h4><p>Quality 发生在你和世界之间的瞬间。</p></div>
        <div class="it"><div class="c">生活</div><h4>极简生活的真实代价</h4><p>困难的不是扔，而是面对扔掉后的空白。</p></div>
        <div class="it"><div class="c">生活</div><h4>厨房里的系统思维</h4><p>做菜和写代码有惊人的共同点。</p></div>
      </div>
      <div class="dv"></div>
      <div class="cl">
        <div class="it"><div class="c">技术</div><h4>CSS 容器查询改变了一切</h4><p>组件终于可以自己决定长什么样了。</p></div>
        <div class="it"><div class="c">阅读</div><h4>一杯咖啡的等待时间</h4><p>手冲四分钟，刚好做一次正念。</p></div>
        <div class="it"><div class="c">技术</div><h4>静态网站的文艺复兴</h4><p>简单最终胜出，但代价是什么？</p></div>
      </div>
    </div>
    <div class="bot">
      <div class="bot-it"><div class="c">短札</div><h5>学会说"不"的一周</h5></div>
      <div class="bot-it"><div class="c">书摘</div><h5>《人月神话》重读笔记</h5></div>
      <div class="bot-it"><div class="c">工具</div><h5>本月发现的三个好工具</h5></div>
      <div class="bot-it"><div class="c">随想</div><h5>为什么我还在写博客</h5></div>
    </div>
    <footer class="ftr"><span>© 2026 夏天的爱人</span><span>2024 年创刊</span></footer>
  `),
},

// ══════════════ FULL PAGE SET: ml-np-hierarchy ══════════════

// This entry is used as the preview card in the binary choice above
'ml-np-hierarchy': {
  name: '报纸 · 主次双栏',
  subtitle: '左侧大文章+右侧小栏目，主次分明',
  tags: ['hierarchy', 'two-column', 'featured'],
  get html() { return variants['ml-np-h-home'].html; },
},

'ml-np-h-home': {
  name: '首页',
  subtitle: '主次双栏首页，特稿+侧栏文章列表',
  tags: ['home', 'featured', 'hierarchy'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f4f0e8;color:#1a1a1a;min-height:100vh;}
    .hd{text-align:center;padding:20px 40px 16px;border-bottom:4px solid #1a1a1a;max-width:900px;margin:0 auto;}
    .hd h1{font-family:'Playfair Display',serif;font-size:42px;font-weight:900;letter-spacing:-0.02em;line-height:1;}
    .hd .tg{font-size:11px;color:#999;font-style:italic;margin-top:3px;}
    .db{display:flex;justify-content:space-between;max-width:900px;margin:0 auto;padding:8px 40px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;}
    .nav-links{display:flex;gap:16px;}
    .nav-links a{transition:color 0.15s;}
    .nav-links a:hover{color:#1a1a1a;}
    .main{display:grid;grid-template-columns:1.6fr 1px 1fr;gap:0;max-width:900px;margin:0 auto;padding:32px 40px;}
    .main-div{background:#c8c0b4;}
    .main-left{padding-right:32px;}
    .main-left .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.15em;color:#aaa;margin-bottom:8px;}
    .main-left h2{font-size:30px;font-weight:900;line-height:1.2;margin-bottom:12px;}
    .main-left .lead{font-size:16px;line-height:1.85;color:#555;font-style:italic;margin-bottom:16px;}
    .main-left .body{font-size:15px;line-height:1.85;color:#444;}
    .main-left .body p{margin-bottom:14px;text-align:justify;}
    .main-left .read-more{font-family:'Inter',sans-serif;font-size:12px;color:#1a1a1a;border-bottom:1px solid #1a1a1a;padding-bottom:1px;}
    .main-right{padding-left:32px;}
    .main-right .section-title{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;padding-bottom:8px;border-bottom:2px solid #1a1a1a;margin-bottom:12px;}
    .side-item{padding:14px 0;border-bottom:1px solid #ddd5ca;cursor:pointer;transition:all 0.15s;}
    .side-item:hover{padding-left:6px;}
    .side-item:last-child{border-bottom:none;}
    .side-item .c{font-family:'Inter',sans-serif;font-size:8px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:3px;}
    .side-item h4{font-size:14px;font-weight:700;line-height:1.3;margin-bottom:4px;}
    .side-item p{font-size:12px;line-height:1.6;color:#888;}
    .quote{max-width:900px;margin:0 auto;padding:28px 40px;border-top:1px solid #c8c0b4;border-bottom:1px solid #c8c0b4;text-align:center;}
    .quote blockquote{font-family:'Playfair Display',serif;font-size:20px;font-style:italic;color:#5a5048;line-height:1.5;max-width:480px;margin:0 auto;}
    .quote cite{display:block;font-family:'Inter',sans-serif;font-size:10px;color:#aaa;margin-top:8px;font-style:normal;}
    .ftr{max-width:900px;margin:0 auto;padding:16px 40px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="hd"><h1>夏天的爱人</h1><div class="tg">"在字里行间，找到回家的路"</div></header>
    <div class="db"><div class="nav-links"><a href="#">首页</a><a href="#">归档</a><a href="#">关于</a></div><span>2026 年 4 月 14 日</span></div>
    <div class="main">
      <article class="main-left">
        <div class="cat">本期特稿</div>
        <h2>当我们谈论架构时<br>我们谈论什么</h2>
        <p class="lead">软件架构不是一个决定，而是一系列决定的叠加。每一次 trade-off 都在塑造系统的性格。</p>
        <div class="body">
          <p>多数系统设计教程的问题在于：它们从"最佳实践"出发，而不是从约束条件出发。最佳实践是别人的答案，而你需要的是推导出自己答案的方法。</p>
          <p>任何系统设计的第一步不是选技术栈，而是明确约束。约束分三层：业务约束、技术约束和团队约束。最后一个经常被忽略，却往往最致命。</p>
        </div>
        <a class="read-more" href="#">阅读全文 →</a>
      </article>
      <div class="main-div"></div>
      <aside class="main-right">
        <div class="section-title">近期文章</div>
        <div class="side-item"><div class="c">技术</div><h4>函数式编程的哲学根源</h4><p>不可变性不只是技术选择，而是一种世界观。</p></div>
        <div class="side-item"><div class="c">随笔</div><h4>Rust 的第 100 天</h4><p>关于信任一个严格朋友的故事。</p></div>
        <div class="side-item"><div class="c">阅读</div><h4>重读《禅与摩托车维修艺术》</h4><p>Quality 发生在你和世界之间。</p></div>
        <div class="side-item"><div class="c">生活</div><h4>极简生活的代价</h4><p>困难的是面对扔掉后的空白。</p></div>
      </aside>
    </div>
    <div class="quote"><blockquote>"完美不是无可添加，而是无可删减。"</blockquote><cite>— Antoine de Saint-Exupéry</cite></div>
    <footer class="ftr"><span>© 2026</span><span>不定期出刊 · 2024 年创刊</span></footer>
  `),
},

'ml-np-h-article': {
  name: '文章详情页',
  subtitle: '单篇文章阅读体验，宽正文+窄侧栏元信息',
  tags: ['article', 'reading', 'detail'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f4f0e8;color:#1a1a1a;min-height:100vh;}
    .hd{text-align:center;padding:16px 40px 12px;border-bottom:4px solid #1a1a1a;max-width:900px;margin:0 auto;}
    .hd h1{font-family:'Playfair Display',serif;font-size:28px;font-weight:900;letter-spacing:-0.02em;line-height:1;}
    .db{display:flex;justify-content:space-between;max-width:900px;margin:0 auto;padding:8px 40px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;}
    .nav-links{display:flex;gap:16px;}
    .nav-links a{transition:color 0.15s;}
    .nav-links a:hover{color:#1a1a1a;}
    .article-header{max-width:900px;margin:0 auto;padding:40px 40px 0;}
    .article-header .cat{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#aaa;margin-bottom:12px;}
    .article-header h2{font-family:'Playfair Display',serif;font-size:36px;font-weight:900;line-height:1.2;letter-spacing:-0.02em;max-width:640px;}
    .article-header .lead{font-size:18px;line-height:1.8;color:#666;font-style:italic;margin-top:16px;max-width:600px;}
    .article-meta{display:flex;gap:20px;margin-top:20px;padding-bottom:24px;border-bottom:1px solid #c8c0b4;font-family:'Inter',sans-serif;font-size:11px;color:#aaa;}
    .article-body{display:grid;grid-template-columns:1fr 220px;gap:48px;max-width:900px;margin:0 auto;padding:32px 40px;}
    .prose{font-size:16px;line-height:2;color:#333;}
    .prose p{margin-bottom:20px;text-align:justify;}
    .prose h3{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;margin:36px 0 16px;letter-spacing:-0.01em;}
    .prose blockquote{margin:28px 0;padding:20px 24px;border-left:3px solid #1a1a1a;font-style:italic;color:#5a5048;font-size:17px;line-height:1.7;background:#efebe3;}
    .prose code{font-family:'JetBrains Mono',monospace;font-size:14px;background:#e8e3da;padding:2px 6px;border-radius:3px;}
    .prose .separator{text-align:center;color:#c8c0b4;font-size:16px;letter-spacing:0.4em;margin:32px 0;}
    .aside-col{position:sticky;top:32px;align-self:start;}
    .aside-section{margin-bottom:28px;}
    .aside-section h4{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;padding-bottom:6px;border-bottom:2px solid #1a1a1a;margin-bottom:10px;}
    .aside-section .toc-item{display:block;font-family:'Inter',sans-serif;font-size:12px;color:#888;padding:5px 0;border-bottom:1px solid #e8e3da;cursor:pointer;transition:color 0.15s;}
    .aside-section .toc-item:hover{color:#1a1a1a;}
    .aside-section .tag-list{display:flex;flex-wrap:wrap;gap:6px;}
    .aside-section .tag{font-family:'Inter',sans-serif;font-size:10px;padding:4px 10px;border:1px solid #d4cdc4;border-radius:3px;color:#888;cursor:pointer;transition:all 0.15s;}
    .aside-section .tag:hover{border-color:#1a1a1a;color:#1a1a1a;}
    .aside-section .related{font-size:13px;padding:8px 0;border-bottom:1px solid #e8e3da;cursor:pointer;transition:color 0.15s;}
    .aside-section .related:hover{color:#666;}
    .article-nav{display:grid;grid-template-columns:1fr 1fr;gap:24px;max-width:900px;margin:0 auto;padding:24px 40px;border-top:1px solid #c8c0b4;}
    .article-nav a{font-family:'Inter',sans-serif;font-size:12px;color:#888;transition:color 0.15s;}
    .article-nav a:hover{color:#1a1a1a;}
    .article-nav .label{font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:4px;}
    .article-nav .title{font-family:'Noto Serif SC',serif;font-size:14px;font-weight:600;color:#1a1a1a;}
    .article-nav .next{text-align:right;}
    .ftr{max-width:900px;margin:0 auto;padding:16px 40px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="hd"><h1>夏天的爱人</h1></header>
    <div class="db"><div class="nav-links"><a href="#">首页</a><a href="#">归档</a><a href="#">关于</a></div><span>← 返回首页</span></div>
    <div class="article-header">
      <div class="cat">技术 · 系统设计</div>
      <h2>当我们谈论架构时<br>我们谈论什么</h2>
      <p class="lead">软件架构不是一个决定，而是一系列决定的叠加。每一次 trade-off 都在塑造系统的性格。</p>
      <div class="article-meta"><span>2026 年 4 月 10 日</span><span>·</span><span>约 12 分钟阅读</span><span>·</span><span>3,200 字</span></div>
    </div>
    <div class="article-body">
      <article class="prose">
        <p>多数系统设计教程的问题在于：它们从"最佳实践"出发，而不是从约束条件出发。最佳实践是别人的答案，而你需要的是推导出自己答案的方法。</p>
        <h3>从约束开始</h3>
        <p>任何系统设计的第一步不是选技术栈，而是明确约束。约束分三层：<strong>业务约束</strong>（必须满足的需求）、<strong>技术约束</strong>（当前必须面对的限制）和<strong>团队约束</strong>（经常被忽略，却往往最致命）。</p>
        <p>这和厨房里的系统思维异曲同工：你不会在没有看冰箱之前就决定今晚做什么菜。约束决定方案，而不是反过来。</p>
        <blockquote>"好的架构让人忘记它的存在。你走进一座好建筑，感受到的是空间和光线，而不是墙壁和梁柱。"</blockquote>
        <h3>复杂度的代价</h3>
        <p>每一个架构决策都是在简单性和某种能力之间做交换。关键不是避免复杂度，而是确保你在正确的地方引入了复杂度。微服务不一定比单体好——取决于你的团队规模和部署频率。</p>
        <p>我见过太多团队在 DAU 只有几千的产品上搭建了"支撑百万级"的架构。这不是远见，这是浪费。先用 <code>sqlite</code> 跑起来，等真正需要的时候再拆分。</p>
        <div class="separator">· · ·</div>
        <p>架构不是画一张图然后执行。它是一系列持续的、小的、可逆的决策。最好的架构师不是能画最复杂的图的人，而是能把复杂问题拆成简单决策的人。</p>
      </article>
      <aside class="aside-col">
        <div class="aside-section">
          <h4>目录</h4>
          <a class="toc-item" href="#">从约束开始</a>
          <a class="toc-item" href="#">复杂度的代价</a>
        </div>
        <div class="aside-section">
          <h4>标签</h4>
          <div class="tag-list"><span class="tag">系统设计</span><span class="tag">架构</span><span class="tag">第一性原理</span></div>
        </div>
        <div class="aside-section">
          <h4>相关文章</h4>
          <div class="related">厨房里的系统思维</div>
          <div class="related">为什么我不用 ORM</div>
          <div class="related">静态网站的文艺复兴</div>
        </div>
      </aside>
    </div>
    <div class="article-nav">
      <a href="#"><div class="label">← 上一篇</div><div class="title">Rust 的第 100 天</div></a>
      <a class="next" href="#"><div class="label">下一篇 →</div><div class="title">函数式编程的哲学根源</div></a>
    </div>
    <footer class="ftr"><span>© 2026</span><span>不定期出刊 · 2024 年创刊</span></footer>
  `),
},

'ml-np-h-archive': {
  name: '归档页',
  subtitle: '按年月分组的文章索引，全部内容一览',
  tags: ['archive', 'index', 'chronological'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f4f0e8;color:#1a1a1a;min-height:100vh;}
    .hd{text-align:center;padding:16px 40px 12px;border-bottom:4px solid #1a1a1a;max-width:900px;margin:0 auto;}
    .hd h1{font-family:'Playfair Display',serif;font-size:28px;font-weight:900;letter-spacing:-0.02em;line-height:1;}
    .db{display:flex;justify-content:space-between;max-width:900px;margin:0 auto;padding:8px 40px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;}
    .nav-links{display:flex;gap:16px;}
    .nav-links a{transition:color 0.15s;}
    .nav-links a:hover{color:#1a1a1a;}
    .page-header{max-width:900px;margin:0 auto;padding:40px 40px 20px;}
    .page-header h2{font-family:'Playfair Display',serif;font-size:32px;font-weight:900;letter-spacing:-0.02em;}
    .page-header p{font-family:'Inter',sans-serif;font-size:13px;color:#999;margin-top:6px;}
    .archive-filters{max-width:900px;margin:0 auto;padding:0 40px 24px;display:flex;gap:8px;border-bottom:1px solid #c8c0b4;}
    .filter-btn{font-family:'Inter',sans-serif;font-size:11px;padding:5px 12px;border:1px solid #d4cdc4;border-radius:3px;background:transparent;color:#888;cursor:pointer;transition:all 0.15s;}
    .filter-btn:hover,.filter-btn.active{border-color:#1a1a1a;color:#1a1a1a;background:#ece7de;}
    .archive-content{display:grid;grid-template-columns:1fr 200px;gap:40px;max-width:900px;margin:0 auto;padding:32px 40px;}
    .year-group{margin-bottom:36px;}
    .year-label{font-family:'Playfair Display',serif;font-size:48px;font-weight:900;color:#e0d9ce;letter-spacing:-0.03em;margin-bottom:4px;line-height:1;}
    .month-group{margin-bottom:20px;}
    .month-label{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#bbb;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #ddd5ca;}
    .archive-item{display:grid;grid-template-columns:1fr auto;gap:16px;padding:10px 0;border-bottom:1px solid #ece7de;cursor:pointer;transition:all 0.15s;}
    .archive-item:hover{padding-left:6px;}
    .archive-item:last-child{border-bottom:none;}
    .archive-item .title{font-size:15px;font-weight:600;line-height:1.35;}
    .archive-item .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.08em;color:#bbb;margin-top:2px;}
    .archive-item .date{font-family:'Inter',sans-serif;font-size:11px;color:#bbb;white-space:nowrap;padding-top:2px;}
    .stats-sidebar{position:sticky;top:32px;align-self:start;}
    .stats-block{margin-bottom:28px;}
    .stats-block h4{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;padding-bottom:6px;border-bottom:2px solid #1a1a1a;margin-bottom:10px;}
    .stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #ece7de;font-family:'Inter',sans-serif;font-size:12px;color:#888;}
    .stat-row .count{font-weight:600;color:#1a1a1a;}
    .tag-cloud{display:flex;flex-wrap:wrap;gap:4px;}
    .tag-cloud .t{font-family:'Inter',sans-serif;font-size:10px;padding:3px 8px;border:1px solid #d4cdc4;border-radius:3px;color:#999;cursor:pointer;transition:all 0.15s;}
    .tag-cloud .t:hover{border-color:#1a1a1a;color:#1a1a1a;}
    .ftr{max-width:900px;margin:0 auto;padding:16px 40px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="hd"><h1>夏天的爱人</h1></header>
    <div class="db"><div class="nav-links"><a href="#">首页</a><a href="#" style="color:#1a1a1a;">归档</a><a href="#">关于</a></div><span>共 42 篇</span></div>
    <div class="page-header">
      <h2>归档</h2>
      <p>所有文章，按时间排列。从 2024 年写到现在。</p>
    </div>
    <div class="archive-filters">
      <button class="filter-btn active">全部</button>
      <button class="filter-btn">技术</button>
      <button class="filter-btn">随笔</button>
      <button class="filter-btn">阅读</button>
      <button class="filter-btn">生活</button>
    </div>
    <div class="archive-content">
      <div>
        <div class="year-group">
          <div class="year-label">2026</div>
          <div class="month-group">
            <div class="month-label">四月 · 3 篇</div>
            <div class="archive-item"><div><div class="title">当我们谈论架构时我们谈论什么</div><div class="cat">技术</div></div><div class="date">04.10</div></div>
            <div class="archive-item"><div><div class="title">Rust 的第 100 天</div><div class="cat">随笔</div></div><div class="date">04.08</div></div>
            <div class="archive-item"><div><div class="title">为什么好的 API 像好的散文</div><div class="cat">技术</div></div><div class="date">04.02</div></div>
          </div>
          <div class="month-group">
            <div class="month-label">三月 · 4 篇</div>
            <div class="archive-item"><div><div class="title">重读《禅与摩托车维修艺术》</div><div class="cat">阅读</div></div><div class="date">03.22</div></div>
            <div class="archive-item"><div><div class="title">厨房里的系统思维</div><div class="cat">生活</div></div><div class="date">03.18</div></div>
            <div class="archive-item"><div><div class="title">极简生活的真实代价</div><div class="cat">随笔</div></div><div class="date">03.10</div></div>
            <div class="archive-item"><div><div class="title">CSS 容器查询改变了一切</div><div class="cat">技术</div></div><div class="date">03.05</div></div>
          </div>
          <div class="month-group">
            <div class="month-label">二月 · 3 篇</div>
            <div class="archive-item"><div><div class="title">为什么我不用 ORM</div><div class="cat">技术</div></div><div class="date">02.20</div></div>
            <div class="archive-item"><div><div class="title">一杯咖啡的等待时间</div><div class="cat">生活</div></div><div class="date">02.12</div></div>
            <div class="archive-item"><div><div class="title">静态网站的文艺复兴</div><div class="cat">技术</div></div><div class="date">02.05</div></div>
          </div>
        </div>
        <div class="year-group">
          <div class="year-label">2025</div>
          <div class="month-group">
            <div class="month-label">十二月 · 2 篇</div>
            <div class="archive-item"><div><div class="title">年末：写给代码和自己的信</div><div class="cat">随笔</div></div><div class="date">12.28</div></div>
            <div class="archive-item"><div><div class="title">函数式编程的哲学根源</div><div class="cat">技术</div></div><div class="date">12.15</div></div>
          </div>
        </div>
      </div>
      <aside class="stats-sidebar">
        <div class="stats-block">
          <h4>统计</h4>
          <div class="stat-row"><span>文章总数</span><span class="count">42</span></div>
          <div class="stat-row"><span>总字数</span><span class="count">86,400</span></div>
          <div class="stat-row"><span>写作天数</span><span class="count">128</span></div>
        </div>
        <div class="stats-block">
          <h4>分类</h4>
          <div class="stat-row"><span>技术</span><span class="count">18</span></div>
          <div class="stat-row"><span>随笔</span><span class="count">10</span></div>
          <div class="stat-row"><span>阅读</span><span class="count">8</span></div>
          <div class="stat-row"><span>生活</span><span class="count">6</span></div>
        </div>
        <div class="stats-block">
          <h4>标签</h4>
          <div class="tag-cloud">
            <span class="t">架构</span><span class="t">Rust</span><span class="t">前端</span><span class="t">CSS</span><span class="t">数据库</span><span class="t">书评</span><span class="t">极简</span><span class="t">哲学</span>
          </div>
        </div>
      </aside>
    </div>
    <footer class="ftr"><span>© 2026</span><span>不定期出刊 · 2024 年创刊</span></footer>
  `),
},

'ml-np-h-about': {
  name: '关于页',
  subtitle: '个人介绍、写作理念、联系方式',
  tags: ['about', 'personal', 'bio'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC',Georgia,serif;background:#f4f0e8;color:#1a1a1a;min-height:100vh;}
    .hd{text-align:center;padding:16px 40px 12px;border-bottom:4px solid #1a1a1a;max-width:900px;margin:0 auto;}
    .hd h1{font-family:'Playfair Display',serif;font-size:28px;font-weight:900;letter-spacing:-0.02em;line-height:1;}
    .db{display:flex;justify-content:space-between;max-width:900px;margin:0 auto;padding:8px 40px;border-bottom:2px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;}
    .nav-links{display:flex;gap:16px;}
    .nav-links a{transition:color 0.15s;}
    .nav-links a:hover{color:#1a1a1a;}
    .about-content{display:grid;grid-template-columns:1fr 240px;gap:48px;max-width:900px;margin:0 auto;padding:40px 40px 48px;}
    .about-main h2{font-family:'Playfair Display',serif;font-size:32px;font-weight:900;letter-spacing:-0.02em;margin-bottom:20px;}
    .about-main .intro{font-size:18px;line-height:1.85;color:#555;font-style:italic;margin-bottom:24px;padding-bottom:24px;border-bottom:1px solid #ddd5ca;}
    .about-main h3{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;margin:32px 0 12px;}
    .about-main p{font-size:15px;line-height:1.9;color:#444;margin-bottom:14px;text-align:justify;}
    .about-main .values{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0;}
    .value-card{padding:16px;border:1px solid #ddd5ca;border-radius:4px;}
    .value-card h4{font-family:'Inter',sans-serif;font-size:12px;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;}
    .value-card p{font-size:13px;color:#888;line-height:1.6;margin:0;}
    .about-aside{position:sticky;top:32px;align-self:start;}
    .aside-block{margin-bottom:28px;}
    .aside-block h4{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.12em;color:#aaa;padding-bottom:6px;border-bottom:2px solid #1a1a1a;margin-bottom:10px;}
    .avatar{width:100%;aspect-ratio:1;border-radius:4px;background:linear-gradient(135deg,#d4cdc4,#c8c0b4);margin-bottom:12px;display:flex;align-items:center;justify-content:center;font-size:48px;}
    .contact-item{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #ece7de;font-family:'Inter',sans-serif;font-size:12px;}
    .contact-item .label{color:#aaa;}
    .contact-item .value{color:#1a1a1a;font-weight:500;}
    .colophon{margin-top:8px;font-family:'Inter',sans-serif;font-size:11px;color:#aaa;line-height:1.7;}
    .ftr{max-width:900px;margin:0 auto;padding:16px 40px;border-top:4px solid #1a1a1a;font-family:'Inter',sans-serif;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:0.12em;display:flex;justify-content:space-between;}
  `, `
    <header class="hd"><h1>夏天的爱人</h1></header>
    <div class="db"><div class="nav-links"><a href="#">首页</a><a href="#">归档</a><a href="#" style="color:#1a1a1a;">关于</a></div><span></span></div>
    <div class="about-content">
      <main class="about-main">
        <h2>关于这里</h2>
        <p class="intro">这是一份个人刊物——记录代码背后的思考、阅读之后的余韵、以及生活中那些不值一提却值得记住的瞬间。</p>

        <h3>关于我</h3>
        <p>我是一个写代码的人，也是一个用文字思考的人。白天在键盘上写程序，晚上（偶尔）在键盘上写文字。相信技术不只是工具，也是一种表达方式。</p>
        <p>工作中主要做全栈开发，对系统设计、Rust、前端架构比较感兴趣。业余时间读书、做菜、拍照、偶尔发呆。</p>

        <h3>为什么写</h3>
        <p>写作是最好的思考工具。很多时候，我以为自己理解了一个概念，直到试图把它写下来——然后发现理解里全是漏洞。</p>
        <p>这个网站没有 KPI、没有更新频率、没有流量目标。它只是一个地方，用来放那些值得被认真写下来的东西。</p>

        <div class="values">
          <div class="value-card"><h4>精确</h4><p>宁可少写，也不含糊。每个词都应该准确地表达意图。</p></div>
          <div class="value-card"><h4>诚实</h4><p>不懂就说不懂，不确定就标注为假设。</p></div>
          <div class="value-card"><h4>留白</h4><p>好的文章不是写满的，而是给读者留了思考空间。</p></div>
          <div class="value-card"><h4>持久</h4><p>不追热点，写在三年后仍然有价值的东西。</p></div>
        </div>

        <h3>版权声明</h3>
        <p>除非特别标注，所有文章均为原创。你可以自由引用，但请注明出处。如果某篇文章帮到了你，告诉我会让我很开心。</p>
      </main>
      <aside class="about-aside">
        <div class="aside-block">
          <div class="avatar">🖊</div>
        </div>
        <div class="aside-block">
          <h4>联系</h4>
          <div class="contact-item"><span class="label">GitHub</span><span class="value">@xiatian</span></div>
          <div class="contact-item"><span class="label">Twitter</span><span class="value">@xiatian_dev</span></div>
          <div class="contact-item"><span class="label">Email</span><span class="value">hi@xiatian.dev</span></div>
          <div class="contact-item"><span class="label">RSS</span><span class="value">/feed.xml</span></div>
        </div>
        <div class="aside-block">
          <h4>关于本站</h4>
          <p class="colophon">使用 Astro 构建<br>部署于 GitHub Pages<br>字体：Playfair Display + Noto Serif SC + Inter<br>设计灵感：传统报纸排版<br>2024 年创刊</p>
        </div>
      </aside>
    </div>
    <footer class="ftr"><span>© 2026</span><span>不定期出刊 · 2024 年创刊</span></footer>
  `),
},

// ══════════════ ROUND 4: literary refinements ══════════════

'ml-lt-warm': {
  name: '文艺 · 暖纸张',
  subtitle: '米黄底色、怀旧纸张感、温柔的衬线',
  tags: ['warm', 'paper', 'nostalgic'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC','Lora',Georgia,serif;background:#f5efe5;color:#3a3128;min-height:100vh;}
    .wn-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 48px;max-width:700px;margin:0 auto;}
    .wn-brand{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;color:#3a3128;}
    .wn-links{display:flex;gap:20px;font-family:'Inter',sans-serif;font-size:12px;color:#b8a890;}
    .wn-hero{max-width:700px;margin:0 auto;padding:72px 48px 44px;}
    .wn-hero .issue{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.14em;color:#c4b49a;margin-bottom:16px;}
    .wn-hero h1{font-family:'Playfair Display',serif;font-size:36px;font-weight:700;line-height:1.3;letter-spacing:-0.01em;color:#2c2218;}
    .wn-hero .lead{font-size:17px;line-height:1.9;color:#7a6d5e;margin-top:16px;font-style:italic;}
    .wn-hero .read{display:inline-block;margin-top:20px;font-family:'Inter',sans-serif;font-size:12px;color:#3a3128;border-bottom:1px solid #3a3128;padding-bottom:2px;}
    .wn-sep{max-width:700px;margin:0 auto;padding:0 48px;}
    .wn-sep .ornament{text-align:center;color:#d4c4aa;font-size:18px;padding:16px 0;letter-spacing:0.3em;}
    .wn-list{max-width:700px;margin:0 auto;padding:0 48px 40px;}
    .wn-item{padding:22px 0;border-bottom:1px solid #e8dece;cursor:pointer;transition:all 0.2s;}
    .wn-item:hover{padding-left:8px;}
    .wn-item .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#c4b49a;margin-bottom:4px;}
    .wn-item h3{font-size:18px;font-weight:600;line-height:1.35;color:#2c2218;margin-bottom:4px;}
    .wn-item p{font-size:14px;color:#8a7d6e;line-height:1.65;}
    .wn-item .date{font-family:'Inter',sans-serif;font-size:11px;color:#c4b49a;margin-top:6px;}
    .wn-footer{max-width:700px;margin:0 auto;padding:36px 48px;border-top:1px solid #ddd0be;text-align:center;font-family:'Inter',sans-serif;font-size:10px;color:#c4b49a;}
  `, `
    <nav class="wn-nav"><div class="wn-brand">夏天的爱人</div><div class="wn-links"><a href="#">文章</a><a href="#">归档</a><a href="#">关于</a></div></nav>
    <section class="wn-hero">
      <div class="issue">第十二期 · 2026 年 4 月</div>
      <h1>在代码与诗意之间，<br>寻找表达的可能</h1>
      <p class="lead">技术写作不应该只是教程。它可以是一种审美实践——关于如何用精确的语言，描述模糊的直觉。</p>
      <a class="read" href="#">阅读全文 →</a>
    </section>
    <div class="wn-sep"><div class="ornament">· · ·</div></div>
    <section class="wn-list">
      <div class="wn-item"><div class="cat">技术</div><h3>当我们谈论架构时我们谈论什么</h3><p>软件架构不是一个决定，而是一系列决定的叠加。</p><div class="date">四月 10 日</div></div>
      <div class="wn-item"><div class="cat">随笔</div><h3>Rust 的第 100 天</h3><p>关于信任一个严格但正确的朋友的故事。</p><div class="date">四月 2 日</div></div>
      <div class="wn-item"><div class="cat">阅读</div><h3>重读《禅与摩托车维修艺术》</h3><p>Quality 发生在你和世界之间的那个瞬间。</p><div class="date">三月 22 日</div></div>
      <div class="wn-item"><div class="cat">生活</div><h3>极简生活的真实代价</h3><p>扔掉东西是容易的。困难的是面对之后的空白。</p><div class="date">三月 15 日</div></div>
    </section>
    <footer class="wn-footer">夏天的爱人 · 一份关于代码与生活的个人刊物 · 2024 年创刊</footer>
  `),
},

'ml-lt-cool': {
  name: '文艺 · 冷纸张',
  subtitle: '灰白底色、现代感、冷静克制的优雅',
  tags: ['cool', 'modern', 'restrained'],
  html: () => wrapPage(`
    body{font-family:'Noto Serif SC','Lora',Georgia,serif;background:#f0f0ee;color:#2a2a2a;min-height:100vh;}
    .cn-nav{display:flex;justify-content:space-between;align-items:center;padding:24px 48px;max-width:700px;margin:0 auto;}
    .cn-brand{font-family:'Playfair Display',serif;font-size:18px;font-weight:700;letter-spacing:-0.02em;color:#1a1a1a;}
    .cn-links{display:flex;gap:20px;font-family:'Inter',sans-serif;font-size:12px;color:#aaa;}
    .cn-hero{max-width:700px;margin:0 auto;padding:72px 48px 48px;}
    .cn-hero .issue{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.15em;color:#bbb;margin-bottom:16px;}
    .cn-hero h1{font-family:'Playfair Display',serif;font-size:34px;font-weight:700;line-height:1.3;letter-spacing:-0.01em;color:#1a1a1a;}
    .cn-hero .lead{font-size:17px;line-height:1.9;color:#777;margin-top:16px;font-style:italic;}
    .cn-hero .read{display:inline-block;margin-top:20px;font-family:'Inter',sans-serif;font-size:12px;color:#1a1a1a;border-bottom:1px solid #1a1a1a;padding-bottom:2px;letter-spacing:0.02em;}
    .cn-sep{max-width:700px;margin:0 auto;padding:0 48px;}
    .cn-sep hr{border:none;height:1px;background:#ddd;}
    .cn-list{max-width:700px;margin:0 auto;padding:36px 48px;}
    .cn-label{font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#bbb;margin-bottom:20px;}
    .cn-item{padding:20px 0;border-bottom:1px solid #e4e4e2;cursor:pointer;transition:all 0.2s;}
    .cn-item:hover{padding-left:8px;}
    .cn-item .cat{font-family:'Inter',sans-serif;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin-bottom:4px;}
    .cn-item h3{font-size:18px;font-weight:600;line-height:1.35;color:#1a1a1a;margin-bottom:4px;}
    .cn-item p{font-size:14px;color:#888;line-height:1.6;}
    .cn-item .date{font-family:'Inter',sans-serif;font-size:11px;color:#ccc;margin-top:6px;}
    .cn-footer{max-width:700px;margin:0 auto;padding:40px 48px;border-top:1px solid #ddd;text-align:center;font-family:'Inter',sans-serif;font-size:10px;color:#bbb;}
  `, `
    <nav class="cn-nav"><div class="cn-brand">夏天的爱人</div><div class="cn-links"><a href="#">文章</a><a href="#">归档</a><a href="#">关于</a></div></nav>
    <section class="cn-hero">
      <div class="issue">第十二期 · 2026 年 4 月</div>
      <h1>在代码与诗意之间，<br>寻找表达的可能</h1>
      <p class="lead">技术写作不应该只是教程。它可以是一种审美实践——关于如何用精确的语言，描述模糊的直觉。</p>
      <a class="read" href="#">阅读全文 →</a>
    </section>
    <div class="cn-sep"><hr></div>
    <section class="cn-list">
      <div class="cn-label">更多文章</div>
      <div class="cn-item"><div class="cat">技术</div><h3>当我们谈论架构时我们谈论什么</h3><p>软件架构不是一个决定，而是一系列决定的叠加。</p><div class="date">Apr 10</div></div>
      <div class="cn-item"><div class="cat">随笔</div><h3>Rust 的第 100 天</h3><p>关于信任一个严格但正确的朋友的故事。</p><div class="date">Apr 2</div></div>
      <div class="cn-item"><div class="cat">阅读</div><h3>重读《禅与摩托车维修艺术》</h3><p>Quality 发生在你和世界之间的那个瞬间。</p><div class="date">Mar 22</div></div>
      <div class="cn-item"><div class="cat">生活</div><h3>极简生活的真实代价</h3><p>扔掉东西是容易的。困难的是面对之后的空白。</p><div class="date">Mar 15</div></div>
    </section>
    <footer class="cn-footer">夏天的爱人 · 2024 年创刊</footer>
  `),
},

// ── DevNotes → 暗色 vs 亮色 ──
'devnotes-dark': {
  name: '笔记 · 暗色终端',
  subtitle: '深色背景、终端风、代码沉浸',
  tags: ['dark', 'terminal', 'hacker'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#0e0e10;color:#c8c8d0;min-height:100vh;}
    .dn-nav{display:flex;justify-content:space-between;align-items:center;padding:16px 32px;max-width:760px;margin:0 auto;border-bottom:1px solid #1e1e24;}
    .dn-brand{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:#a78bfa;}
    .dn-links{display:flex;gap:16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#555;}
    .dn-links a:hover{color:#a78bfa;}
    .dn-hero{max-width:760px;margin:0 auto;padding:56px 32px 36px;}
    .dn-hero .tag{font-family:'JetBrains Mono',monospace;font-size:12px;color:#a78bfa;margin-bottom:8px;}
    .dn-hero h1{font-size:26px;font-weight:700;letter-spacing:-0.02em;color:#e8e8f0;line-height:1.3;}
    .dn-hero p{font-size:14px;color:#666;margin-top:10px;line-height:1.7;}
    .dn-posts{max-width:760px;margin:0 auto;padding:0 32px 40px;}
    .dn-label{font-family:'JetBrains Mono',monospace;font-size:11px;color:#444;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
    .dn-label::after{content:'';flex:1;height:1px;background:#1e1e24;}
    .dn-post{padding:16px 0;border-bottom:1px solid #18181c;cursor:pointer;transition:all 0.15s;}
    .dn-post:hover{padding-left:8px;}
    .dn-post h3{font-size:15px;font-weight:600;color:#ddd;margin-bottom:4px;}
    .dn-post .desc{font-size:13px;color:#666;line-height:1.5;}
    .dn-post .meta{display:flex;gap:10px;margin-top:6px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#444;}
    .dn-post .meta .t{color:#a78bfa;background:rgba(167,139,250,0.1);padding:1px 6px;border-radius:3px;}
    .dn-topics{max-width:760px;margin:0 auto;padding:0 32px 40px;display:flex;flex-wrap:wrap;gap:6px;}
    .dn-tp{font-family:'JetBrains Mono',monospace;font-size:11px;padding:4px 10px;border:1px solid #1e1e24;border-radius:4px;color:#555;cursor:pointer;transition:0.15s;}
    .dn-tp:hover{border-color:#a78bfa;color:#a78bfa;}
    .dn-footer{max-width:760px;margin:0 auto;padding:24px 32px;border-top:1px solid #1e1e24;font-family:'JetBrains Mono',monospace;font-size:11px;color:#333;display:flex;justify-content:space-between;}
  `, `
    <nav class="dn-nav"><div class="dn-brand">xiatian_</div><div class="dn-links"><a href="#">posts</a><a href="#">topics</a><a href="#">about</a><a href="#">rss</a></div></nav>
    <section class="dn-hero">
      <div class="tag">// dev notes</div>
      <h1>写代码，也写关于代码的文字</h1>
      <p>技术笔记、读书感悟、偶尔的生活碎片。写给半年后的自己。</p>
    </section>
    <div class="dn-posts">
      <div class="dn-label">recent</div>
      <div class="dn-post"><h3>系统设计的第一性原理</h3><p class="desc">不从最佳实践出发，从约束条件推导。</p><div class="meta"><span>2026-04-10</span><span class="t">systems</span><span>8 min</span></div></div>
      <div class="dn-post"><h3>Rust 所有权模型的心智模型</h3><p class="desc">不讲语法，只讲直觉。三个类比帮你理解 borrow checker。</p><div class="meta"><span>2026-04-02</span><span class="t">rust</span><span>12 min</span></div></div>
      <div class="dn-post"><h3>为什么我不用 ORM</h3><p class="desc">真正的痛点不是写 SQL，而是管理 schema 和迁移。</p><div class="meta"><span>2026-03-22</span><span class="t">database</span><span>6 min</span></div></div>
      <div class="dn-post"><h3>CSS 容器查询改变了一切</h3><p class="desc">组件可以自己决定长什么样了。这才是真正的组件化。</p><div class="meta"><span>2026-03-15</span><span class="t">css</span><span>5 min</span></div></div>
    </div>
    <div class="dn-posts"><div class="dn-label">topics</div></div>
    <div class="dn-topics"><span class="dn-tp">rust (8)</span><span class="dn-tp">systems (6)</span><span class="dn-tp">frontend (11)</span><span class="dn-tp">database (4)</span><span class="dn-tp">css (5)</span><span class="dn-tp">reading (7)</span></div>
    <footer class="dn-footer"><span>© 2026 xiatian</span><span>built with astro</span></footer>
  `),
},

'devnotes-light': {
  name: '笔记 · 亮色清爽',
  subtitle: '白底、代码友好但阳光感、适合长时间阅读',
  tags: ['light', 'clean', 'readable'],
  html: () => wrapPage(`
    body{font-family:'Inter',sans-serif;background:#fdfdfc;color:#1a1a1a;min-height:100vh;}
    .dl-nav{display:flex;justify-content:space-between;align-items:center;padding:20px 32px;max-width:760px;margin:0 auto;border-bottom:1px solid #eee;}
    .dl-brand{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;}
    .dl-brand .accent{color:#6366f1;}
    .dl-links{display:flex;gap:16px;font-size:12px;color:#aaa;}
    .dl-links a:hover{color:#1a1a1a;}
    .dl-hero{max-width:760px;margin:0 auto;padding:56px 32px 36px;}
    .dl-hero .hi{font-family:'JetBrains Mono',monospace;font-size:13px;color:#6366f1;margin-bottom:8px;}
    .dl-hero h1{font-size:26px;font-weight:800;letter-spacing:-0.03em;line-height:1.3;}
    .dl-hero p{font-size:14px;color:#888;margin-top:10px;line-height:1.7;max-width:480px;}
    .dl-section{max-width:760px;margin:0 auto;padding:0 32px 40px;}
    .dl-label{font-family:'JetBrains Mono',monospace;font-size:11px;color:#ccc;margin-bottom:12px;display:flex;align-items:center;gap:8px;text-transform:uppercase;letter-spacing:0.06em;}
    .dl-label::after{content:'';flex:1;height:1px;background:#f0f0f0;}
    .dl-post{padding:16px 20px;margin-bottom:8px;border-radius:8px;border:1px solid #f0f0f0;cursor:pointer;transition:all 0.15s;}
    .dl-post:hover{border-color:#e0e0ff;background:#fafaff;transform:translateX(2px);}
    .dl-post h3{font-size:15px;font-weight:600;margin-bottom:4px;}
    .dl-post .desc{font-size:13px;color:#999;line-height:1.5;}
    .dl-post .meta{display:flex;gap:10px;margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#ccc;}
    .dl-post .meta .t{color:#6366f1;background:#f0f0ff;padding:2px 8px;border-radius:4px;font-weight:500;}
    .dl-topics{max-width:760px;margin:0 auto;padding:0 32px 40px;display:flex;flex-wrap:wrap;gap:6px;}
    .dl-tp{font-family:'JetBrains Mono',monospace;font-size:11px;padding:6px 12px;border:1px solid #eee;border-radius:6px;color:#aaa;cursor:pointer;transition:0.15s;}
    .dl-tp:hover{border-color:#6366f1;color:#6366f1;background:#fafaff;}
    .dl-footer{max-width:760px;margin:0 auto;padding:24px 32px;border-top:1px solid #eee;font-family:'JetBrains Mono',monospace;font-size:11px;color:#ddd;display:flex;justify-content:space-between;}
  `, `
    <nav class="dl-nav"><div class="dl-brand">xiatian<span class="accent">.</span>dev</div><div class="dl-links"><a href="#">文章</a><a href="#">主题</a><a href="#">关于</a><a href="#">RSS</a></div></nav>
    <section class="dl-hero">
      <div class="hi">// hello, world</div>
      <h1>写代码，也写关于代码的文字</h1>
      <p>技术笔记、读书感悟、偶尔的生活碎片。所有文章都是写给半年后的自己看的。</p>
    </section>
    <div class="dl-section">
      <div class="dl-label">recent posts</div>
      <div class="dl-post"><h3>系统设计的第一性原理</h3><p class="desc">不从最佳实践出发，从约束条件推导出自己的答案。</p><div class="meta"><span>2026-04-10</span><span class="t">systems</span><span>8 min</span></div></div>
      <div class="dl-post"><h3>Rust 所有权模型的心智模型</h3><p class="desc">不讲语法，只讲直觉。三个类比帮你理解 borrow checker。</p><div class="meta"><span>2026-04-02</span><span class="t">rust</span><span>12 min</span></div></div>
      <div class="dl-post"><h3>为什么我不用 ORM</h3><p class="desc">真正的痛点不是写 SQL，而是管理 schema 和迁移。</p><div class="meta"><span>2026-03-22</span><span class="t">database</span><span>6 min</span></div></div>
      <div class="dl-post"><h3>CSS 容器查询改变了一切</h3><p class="desc">组件可以自己决定长什么样了。这才是真正的组件化。</p><div class="meta"><span>2026-03-15</span><span class="t">css</span><span>5 min</span></div></div>
    </div>
    <div class="dl-section"><div class="dl-label">topics</div></div>
    <div class="dl-topics"><span class="dl-tp">rust (8)</span><span class="dl-tp">systems (6)</span><span class="dl-tp">frontend (11)</span><span class="dl-tp">database (4)</span><span class="dl-tp">css (5)</span><span class="dl-tp">reading (7)</span><span class="dl-tp">life (3)</span></div>
    <footer class="dl-footer"><span>© 2026 xiatian</span><span>built with astro</span></footer>
  `),
},

};

// ======================== DECISION TREE ========================

const decisionTree = {
  root: {
    title: '你想要什么样的个人内容站？',
    desc: '4 种完全不同的内容组织方式和视觉语言。选择最有感觉的。',
    type: 'quad',
    choices: ['garden', 'journal', 'magazine', 'devnotes'],
  },
  garden: {
    title: '数字花园 — 组织方式',
    desc: '你喜欢 Notion 式的数据库管理，还是 Wiki 式的内容互联？',
    type: 'binary',
    labelA: 'Notion 数据库',
    labelB: 'Wiki 互联',
    choices: ['garden-notion', 'garden-wiki'],
  },
  journal: {
    title: '手帐日记 — 视觉丰富度',
    desc: '配图和视觉元素多一些，还是只要纯粹的文字？',
    type: 'binary',
    labelA: '图文并茂',
    labelB: '纯粹文字',
    choices: ['journal-rich', 'journal-pure'],
  },
  magazine: {
    title: '独立杂志 — 明暗',
    desc: '深色沉浸的夜间阅读感，还是经典浅色的纸张质感？',
    type: 'binary',
    labelA: '深色沉浸',
    labelB: '经典浅色',
    choices: ['magazine-dark', 'magazine-light'],
  },
  devnotes: {
    title: '开发者笔记 — 明暗',
    desc: '暗色终端风沉浸编码感，还是亮色清爽日间阅读？',
    type: 'binary',
    labelA: '暗色终端',
    labelB: '亮色清爽',
    choices: ['devnotes-dark', 'devnotes-light'],
  },
  // ── magazine-light deep refinements ──
  'magazine-light': {
    title: '经典浅色 — 排版气质',
    desc: '你更喜欢报纸式的密集厚重感，还是文艺杂志的优雅留白？',
    type: 'binary',
    labelA: '报纸厚重',
    labelB: '文艺留白',
    choices: ['ml-newspaper', 'ml-literary'],
  },
  'ml-newspaper': {
    title: '报纸风 — 信息密度',
    desc: '三栏齐平最大化内容，还是主次双栏突出重点文章？',
    type: 'binary',
    labelA: '密集三栏',
    labelB: '主次双栏',
    choices: ['ml-np-dense', 'ml-np-hierarchy'],
  },
  'ml-np-hierarchy': {
    title: '主次双栏 — 页面预览',
    desc: '浏览完整网站的各个页面设计。点击任一页面查看全屏效果。',
    type: 'quad',
    choices: ['ml-np-h-home', 'ml-np-h-article', 'ml-np-h-archive', 'ml-np-h-about'],
  },
  'ml-literary': {
    title: '文艺留白 — 色温',
    desc: '暖色米黄的怀旧纸张感，还是冷灰白的现代克制感？',
    type: 'binary',
    labelA: '暖纸张',
    labelB: '冷纸张',
    choices: ['ml-lt-warm', 'ml-lt-cool'],
  },
};

// ======================== APP STATE ========================

const state = {
  path: [],
  currentNode: 'root',
};

// ======================== RENDER ========================

function render() {
  const node = decisionTree[state.currentNode];

  renderBreadcrumb();

  $('#btn-back').style.display = state.path.length > 0 ? '' : 'none';

  const isLeaf = !node;
  $('#btn-export').style.display = isLeaf ? '' : 'none';

  if (isLeaf) {
    showFinal(state.currentNode);
    return;
  }

  $('#final-overlay').style.display = 'none';
  $('#stage-title').textContent = node.title;
  $('#stage-desc').textContent = node.desc;
  $('#stage-info').style.display = '';

  const main = $('#main-content');
  main.innerHTML = '';

  if (node.type === 'quad') {
    const grid = document.createElement('div');
    grid.className = 'grid-4';
    node.choices.forEach((id) => {
      grid.appendChild(createCard(id));
    });
    main.appendChild(grid);
  } else {
    const wrapper = document.createElement('div');
    wrapper.className = 'grid-2-wrapper';
    const grid = document.createElement('div');
    grid.className = 'grid-2';
    node.choices.forEach((id) => {
      grid.appendChild(createCard(id));
    });
    wrapper.appendChild(grid);
    const vs = document.createElement('div');
    vs.className = 'vs-label';
    vs.textContent = 'VS';
    wrapper.appendChild(vs);
    main.appendChild(wrapper);
  }
}

function createCard(variantId) {
  const v = variants[variantId];
  const card = document.createElement('div');
  card.className = 'variant-card animate-in';
  card.onclick = () => selectVariant(variantId);

  const preview = document.createElement('div');
  preview.className = 'variant-preview';
  const iframe = document.createElement('iframe');
  iframe.setAttribute('sandbox', 'allow-same-origin');
  iframe.srcdoc = v.html();
  iframe.loading = 'lazy';
  preview.appendChild(iframe);

  const info = document.createElement('div');
  info.className = 'variant-info';

  const name = document.createElement('div');
  name.className = 'variant-name';
  name.textContent = v.name;

  const subtitle = document.createElement('div');
  subtitle.className = 'variant-subtitle';
  subtitle.textContent = v.subtitle;

  const tags = document.createElement('div');
  tags.className = 'variant-tags';
  v.tags.forEach(t => {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = t;
    tags.appendChild(tag);
  });

  info.appendChild(name);
  info.appendChild(subtitle);
  info.appendChild(tags);
  card.appendChild(preview);
  card.appendChild(info);
  return card;
}

function selectVariant(id) {
  state.path.push({ id: state.currentNode, name: getNodeLabel(state.currentNode) });
  state.currentNode = id;
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function getNodeLabel(nodeId) {
  if (nodeId === 'root') return '开始';
  return variants[nodeId]?.name || nodeId;
}

function renderBreadcrumb() {
  const bc = $('#breadcrumb');
  bc.innerHTML = '';
  const allItems = [...state.path, { id: state.currentNode, name: getNodeLabel(state.currentNode) }];
  allItems.forEach((item, i) => {
    if (i > 0) {
      const sep = document.createElement('span');
      sep.className = 'separator';
      sep.textContent = '›';
      bc.appendChild(sep);
    }
    const crumb = document.createElement('span');
    crumb.className = 'crumb' + (i === allItems.length - 1 ? ' active' : '');
    crumb.textContent = item.name;
    if (i < allItems.length - 1) {
      crumb.onclick = () => navigateTo(i);
    }
    bc.appendChild(crumb);
  });
}

function navigateTo(pathIndex) {
  state.currentNode = state.path[pathIndex].id;
  state.path = state.path.slice(0, pathIndex);
  render();
}

function goBack() {
  if (state.path.length === 0) return;
  const prev = state.path.pop();
  state.currentNode = prev.id;
  render();
}

// ======================== FINAL ========================

function showFinal(variantId) {
  const v = variants[variantId];
  if (!v) return;

  $('#stage-info').style.display = 'none';
  $('#main-content').innerHTML = '';

  const overlay = $('#final-overlay');
  overlay.style.display = '';

  const pathStr = state.path.map(p => p.name).join(' → ') + ' → ' + v.name;
  $('#final-path').textContent = '决策路径: ' + pathStr;

  const previewContainer = $('#final-preview');
  previewContainer.innerHTML = '';
  const iframe = document.createElement('iframe');
  iframe.srcdoc = v.html();
  iframe.setAttribute('sandbox', 'allow-same-origin');
  previewContainer.appendChild(iframe);
}

function exportHTML() {
  const v = variants[state.currentNode];
  if (!v) return;
  const html = v.html();
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.currentNode}-design.html`;
  a.click();
  URL.revokeObjectURL(url);
}

function fullscreenPreview() {
  const v = variants[state.currentNode];
  if (!v) return;
  const w = window.open('', '_blank');
  w.document.write(v.html());
  w.document.close();
}

// ======================== RANDOM ========================

function randomExplore() {
  state.path = [];
  state.currentNode = 'root';
  let current = 'root';
  while (decisionTree[current]) {
    const node = decisionTree[current];
    const randomChoice = node.choices[Math.floor(Math.random() * node.choices.length)];
    state.path.push({ id: current, name: getNodeLabel(current) });
    current = randomChoice;
  }
  state.currentNode = current;
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ======================== INIT ========================

function init() {
  $('#btn-back').onclick = goBack;
  $('#btn-random').onclick = randomExplore;
  $('#btn-export').onclick = exportHTML;
  $('#btn-restart').onclick = () => {
    state.path = [];
    state.currentNode = 'root';
    $('#final-overlay').style.display = 'none';
    render();
  };
  $('#btn-export-final').onclick = exportHTML;
  $('#btn-fullscreen').onclick = fullscreenPreview;
  render();
}

document.addEventListener('DOMContentLoaded', init);
