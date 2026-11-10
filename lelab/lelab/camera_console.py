"""Browser page for the camera console, served by FastAPI at ``/cameras``.

Kept as a plain HTML string (not part of the React bundle) so camera diagnostics ship
and update with the Python package: no ``npm run build`` step, nothing to remember at
startup — open ``http://<host>:8000/cameras`` whenever a camera is involved.
"""

HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>LeLab 相机控制台</title>
<style>
 body{background:#14161a;color:#e8e8ea;font-family:system-ui,"Noto Sans CJK SC",sans-serif;margin:18px}
 h1{font-size:18px;margin:0 0 6px}
 .hint{color:#9aa0a6;font-size:12px;margin:0 0 14px;line-height:1.6}
 .chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
 .chip{background:#22262d;border:1px solid #38404b;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
 .chip.sel{border-color:#5b9dff;background:#1d2835}
 .chip.bad{border-color:#b3403f}
 .stage{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
 img#feed{width:640px;height:480px;background:#000;border:1px solid #38404b;border-radius:6px}
 .bar{margin:14px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{background:#2b3a52;border:1px solid #4a6488;color:#e8e8ea;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}
 table{border-collapse:collapse;font-size:12px;width:100%}
 th,td{border-bottom:1px solid #2a2f37;padding:5px 8px;text-align:left;vertical-align:top}
 th{color:#9aa0a6;font-weight:600}
 .ok{color:#69d177}.no{color:#ff7b72}
 pre#verdict{background:#1b1f25;border:1px solid #2a2f37;border-radius:6px;padding:10px;font-size:12px;white-space:pre-wrap}
</style></head><body>
<h1>LeLab 相机控制台</h1>
<p class="hint">任意时刻看任意一路画面，不必开始录制。<b>同一时间只开一路大图</b>：UVC 是按相机描述符<b>预约</b>
等时带宽（跟 fourcc／分辨率／帧率无关），能不能同时放下多路取决于主机控制器固件——本机（AMD）实测一颗控制器
只容得下 1 路，别的机器（Intel）可能同时容下 3 路。所以下面用“并发验证”实测，而不是照本宣科。<br>
点相机卡片切换画面；“重新体检”逐只单测；“并发验证”同时开所有路，用来看谁抢不到带宽。</p>
<div class="chips" id="chips"></div>
<div class="stage"><img id="feed" alt="预览"><div><pre id="cur">未选择相机</pre></div></div>
<div class="bar">
  <button onclick="loadHealth(false)">重新体检（逐只单测）</button>
  <button onclick="loadHealth(true)">并发验证（同时全开）</button>
  <span id="stamp" style="color:#9aa0a6;font-size:12px"></span>
</div>
<table id="tbl"></table>
<pre id="verdict"></pre>
<script>
let sel = null, meta = null, timer = null;
function streamSrc(){
  if (!sel) return;
  const img = document.getElementById('feed');
  img.src = '/camera-feed/' + encodeURIComponent(sel) + '?t=' + Date.now();
}
function pick(node, name, ctrl, ok){
  sel = node; meta = {node, name, ctrl, ok};
  document.getElementById('cur').textContent = node + '  ' + name + '\\n控制器 ' + ctrl + (ok ? '\\n状态 正常' : '\\n状态 异常');
  [...document.querySelectorAll('.chip')].forEach(c => c.classList.toggle('sel', c.dataset.node === node));
  streamSrc();
  if (timer) clearInterval(timer);
  // 后端会在 10 分钟后释放设备（避免长期占用相机），这里自动续上
  timer = setInterval(streamSrc, 90000);
}
document.addEventListener('DOMContentLoaded', () => {
  const img = document.getElementById('feed');
  img.onerror = () => { setTimeout(streamSrc, 2000); };
});
function chip(d){
  const el = document.createElement('div');
  el.className = 'chip' + (d.ok === false ? ' bad' : '');
  el.dataset.node = d.node;
  el.textContent = d.node + ' · ' + d.name + ' · ' + (d.fps ? d.fps + 'fps' : d.controller);
  el.title = d.error || '';
  el.onclick = () => pick(d.node, d.name, d.controller, d.ok !== false);
  return el;
}
async function loadHealth(conc){
  const r = await fetch('/camera-health?' + (conc ? 'concurrency=1' : 'refresh=1'));
  const j = await r.json();
  const chips = document.getElementById('chips'); chips.innerHTML = '';
  const rows = j.devices || [];
  rows.forEach(chip);
  if (j.status === 'busy'){ document.getElementById('verdict').textContent = j.message; document.getElementById('tbl').innerHTML=''; if(rows[0]) pick(rows[0].node, rows[0].name, rows[0].controller, false); return; }
  let head = '<tr><th>节点</th><th>名称</th><th>xHCI 控制器</th><th>USB 拓扑</th>';
  let body = '';
  if (j.status === 'concurrency'){
    head += '<th>并发帧数</th><th>并发 fps</th><th>结论</th></tr>';
    (j.concurrent || []).forEach(c => { body += `<tr><td>/dev/video${c.index}</td><td colspan="3"></td><td>${c.frames}</td><td>${c.fps}</td><td class="${c.frames?'ok':'no'}">${c.error || (c.fps ? '出帧正常' : '抢不到带宽')}</td></tr>`; });
  } else {
    head += '<th>单测 fps</th><th>结论</th></tr>';
    rows.forEach(d => { body += `<tr><td>${d.node}</td><td>${d.name}</td><td>${d.controller}</td><td>${d.topology}</td><td class="${d.ok?'ok':'no'}">${d.fps}</td><td class="${d.ok?'ok':'no'}">${d.error || '正常'}</td></tr>`; });
  }
  document.getElementById('tbl').innerHTML = head + body;
  const extra = [];
  if (j.verdict) extra.push(j.verdict);
  if (j.kernel_bandwidth_refusals) extra.push(`内核近 2 分钟记录 ${j.kernel_bandwidth_refusals} 次 “Not enough bandwidth”（等时带宽分配被拒的直接证据）。`);
  document.getElementById('verdict').textContent = extra.join('\\n\\n');
  document.getElementById('stamp').textContent = (j.status === 'concurrency' ? '并发验证完成 ' : '体检完成 ') + new Date().toLocaleTimeString();
  if (!sel && rows[0]) pick(rows[0].node, rows[0].name, rows[0].controller, rows[0].ok !== false);
}
loadHealth(false);
</script></body></html>
"""
