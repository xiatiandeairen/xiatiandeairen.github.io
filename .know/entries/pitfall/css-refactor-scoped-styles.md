# CSS 重构必须覆盖 .astro scoped style

## Symptoms
S1 拆分 global.css 为 6 个模块后，deep review 发现 .astro 文件中的
scoped <style> 块仍有 38 处硬编码间距 + 12 处硬编码 font-size，
完全没有迁移到 token 体系。

## Root cause
重构只关注了全局 CSS 文件（src/styles/*.css），
忽略了页面级 scoped style（archive.astro, series.astro, notes/[slug].astro 等）。
Astro 的 scoped style 不在 global.css 中，容易遗漏。

## Lesson
CSS 架构重构的审计范围必须包括：
1. src/styles/*.css（全局模块）
2. src/pages/**/*.astro 中的 <style> 块
3. src/components/**/*.astro 中的 <style> 块
4. src/layouts/**/*.astro 中的 <style> 块
用 `grep -rn 'font-size: 0\.' src/` 和 `grep -rn 'padding:.*px' src/` 全量扫描。
