/* Dependency-free site checks: node scripts/test_net_center_site.cjs */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'../docs');
const m=require(path.join(root,'assets/js/network.js'));
const visuals=require(path.join(root,'assets/js/visuals.js'));
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
assert.equal(m.D[0][7],600);
assert.deepEqual(m.exact(1),{sites:[2],cost:2580});
assert.deepEqual(m.exact(2),{sites:[0,6],cost:1450});
assert.deepEqual(m.exact(2,'demand'),{sites:[4,9],cost:1550});
assert.deepEqual(m.vertex(),{node:3,radius:360});
assert.deepEqual(m.vertex(true),{node:3,radius:375});
assert.deepEqual(m.absolute(),{edge:3,t:70,radius:300});
assert.deepEqual(m.absolute(true),{edge:3,t:70.5,radius:315.5});
assert.deepEqual(m.curve().map(x=>x.cost),[2580,1450,730,230]);
assert.equal(m.balanced(m.curve()).k,3);
// Check all browser weight values and both candidate domains; cost monotonicity,
// assigned distances and domain membership must hold for the exhaustive curves.
let checked=0;
for(let weight=1;weight<=12;weight++)for(const domain of ['vertices','demand']){
 const w=m.weights.slice();w[2]=weight;const points=m.curve(domain,w);
 points.forEach((p,i)=>{
  assert.equal(p.sites.length,i+1);assert.equal(new Set(p.sites).size,i+1);
  assert.ok(domain!=='demand'||p.sites.every(s=>m.demand.includes(s)));
  if(i)assert.ok(p.cost<=points[i-1].cost);
  const assigned=m.assignments(p.sites);const cost=m.demand.reduce((total,q,j)=>total+w[j]*m.D[q][p.sites[assigned[j]]],0);
  assert.equal(cost,p.cost);checked++;
 });
}
// Independent dense sampling is not the continuous proof, but detects errors in the sweep.
for(const offsets of [false,true]){
 let sampled=Infinity;
 m.edges.forEach(([u,v])=>{const L=m.length(u,v);for(let i=0;i<=4000;i++){const t=L*i/4000;const r=Math.max(...m.demand.map((q,j)=>Math.min(m.D[q][u]+t,m.D[q][v]+L-t)+(offsets?m.offsets[j]:0)));sampled=Math.min(sampled,r);}});
 const a=m.absolute(offsets);assert.ok(a.radius<=sampled+1e-9&&sampled-a.radius<=0.04);
}
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);assert.equal(ids.length,new Set(ids).size);
for(const [,id] of html.matchAll(/href="#([^"]+)"/g))assert.ok(ids.includes(id),`Missing anchor ${id}`);
for(const [,rel] of html.matchAll(/(?:href|src)="(assets\/[^"#]+)"/g))assert.ok(fs.statSync(path.join(root,rel)).isFile(),`Missing asset ${rel}`);
assert.ok(fs.existsSync(path.join(root,'.nojekyll')));assert.ok(!html.includes('@@'));
assert.ok(html.includes(visuals.network({snaps:true,title:'Raw demand and nearest-edge attachment'})));
assert.ok(html.includes(visuals.envelope()));assert.ok(html.includes(visuals.bars(m.curve().map(x=>x.cost),['k=1','k=2','k=3','k=4'],{best:1,title:'Exact median curve'})));
console.log(`PASS: ${checked} finite-site configurations, minimax sampling, fixed expected results, static SVG snapshots and local navigation/assets`);
