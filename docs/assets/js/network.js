/* Small deterministic teaching model. Not a port of the Python solver. */
(function (root) {
  'use strict';
  const nodes = [[30,110],[130,110],[260,110],[390,110],[510,110],[130,30],[390,230],[510,230],[260,230],[80,110]];
  const names = 'ABCDEFGHIJ'.split('');
  const edges = [[0,9],[9,1],[1,2],[2,3],[3,4],[1,5],[3,6],[6,7],[2,8],[8,6]];
  const demand = [0,5,4,7,8,9];
  const weights = [2,1,3,2,2,1];
  const offsets = [15,14,20,16,15,30];
  const raw = [[30,125],[116,30],[510,90],[526,230],[260,245],[80,80]];
  const length = (u,v) => Math.hypot(nodes[u][0]-nodes[v][0],nodes[u][1]-nodes[v][1]);
  const D = nodes.map((_,i)=>nodes.map((_,j)=>i===j?0:Infinity));
  edges.forEach(([u,v])=>{D[u][v]=D[v][u]=length(u,v)});
  for(let k=0;k<nodes.length;k++)for(let i=0;i<nodes.length;i++)for(let j=0;j<nodes.length;j++)D[i][j]=Math.min(D[i][j],D[i][k]+D[k][j]);
  const medianCost=(sites,w=weights)=>demand.reduce((s,q,i)=>s+w[i]*Math.min(...sites.map(v=>D[q][v])),0);
  const vertexRadius=(v,withOffsets=false)=>Math.max(...demand.map((q,i)=>D[q][v]+(withOffsets?offsets[i]:0)));
  function exact(k,domain='vertices',w=weights){
    const pool=domain==='demand'?[...new Set(demand)].sort((a,b)=>a-b):nodes.map((_,i)=>i);
    if(!Number.isInteger(k)||k<1||k>pool.length)throw Error('Invalid site count');
    let best=null;
    function visit(start,chosen){if(chosen.length===k){const cost=medianCost(chosen,w);if(!best||cost<best.cost-1e-9)best={sites:[...chosen],cost};return;}for(let j=start;j<=pool.length-(k-chosen.length);j++)visit(j+1,[...chosen,pool[j]]);}
    visit(0,[]);return best;
  }
  function vertex(withOffsets=false){let best={node:0,radius:vertexRadius(0,withOffsets)};nodes.forEach((_,i)=>{const r=vertexRadius(i,withOffsets);if(r<best.radius-1e-9)best={node:i,radius:r};});return best;}
  function absolute(withOffsets=false){
    const v=vertex(withOffsets);let best={node:v.node,radius:v.radius};
    edges.forEach(([u,w],edge)=>{
      const L=length(u,w),a=demand.map((q,i)=>D[q][u]+(withOffsets?offsets[i]:0)),b=demand.map((q,i)=>D[q][w]+(withOffsets?offsets[i]:0));
      const switches=a.map((x,i)=>Math.max(0,Math.min(L,(b[i]-x+L)/2)));
      const cuts=[...new Set([0,...switches,L])].sort((x,y)=>x-y);
      const evaluate=t=>{const r=Math.max(...a.map((x,i)=>Math.min(x+t,b[i]+L-t)));if(r<best.radius-1e-9)best={edge,t,radius:r};};
      cuts.forEach(evaluate);
      for(let i=0;i<cuts.length-1;i++){
        const lo=cuts[i],hi=cuts[i+1],mid=(lo+hi)/2;
        let A=-Infinity,B=-Infinity;
        switches.forEach((s,j)=>{if(mid<s)A=Math.max(A,a[j]);else B=Math.max(B,b[j]);});
        if(Number.isFinite(A)&&Number.isFinite(B))evaluate(Math.max(lo,Math.min(hi,(B+L-A)/2)));
      }
    });return best;
  }
  function location(result){if(result.node!==undefined)return nodes[result.node];const [u,v]=edges[result.edge],f=result.t/length(u,v);return nodes[u].map((x,i)=>x+f*(nodes[v][i]-x));}
  function assignments(sites){return demand.map(q=>sites.reduce((b,v,i)=>D[q][v]<D[q][sites[b]]-1e-9?i:b,0));}
  function curve(domain='vertices',w=weights,cap=4){return Array.from({length:cap},(_,i)=>exact(i+1,domain,w));}
  function balanced(points,penalty=1){const baseline=points[0].cost;return points.reduce((best,p,i)=>{const score=(baseline?p.cost/baseline:0)+penalty*i/(demand.length-1);return score<best.score-1e-12?{k:i+1,score}:best;},{k:1,score:baseline?1:0});}
  const api={nodes,names,edges,demand,weights,offsets,raw,length,D,medianCost,vertexRadius,exact,vertex,absolute,location,assignments,curve,balanced};
  if(typeof module==='object'&&module.exports)module.exports=api;else root.NetCenterDemo=api;
})(typeof globalThis==='object'?globalThis:this);
