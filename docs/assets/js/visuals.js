(function(root){
  'use strict';
  const m=typeof module==='object'&&module.exports?require('./network.js'):root.NetCenterDemo;
  const colors=['#2478bd','#d47714','#26945d','#8b63c7'];
  const num=x=>Number(x.toFixed(2));
  const line=(a,b,cls,extra='')=>`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" class="${cls}" ${extra}/>`;
  const text=(x,y,s,cls='label',extra='')=>`<text x="${x}" y="${y}" class="${cls}" ${extra}>${s}</text>`;
  const wrap=(title,body,view='0 0 570 280')=>`<svg class="diagram" viewBox="${view}" role="img" aria-label="${title}" xmlns="http://www.w3.org/2000/svg"><title>${title}</title>${body}</svg>`;
  const diamond=(p,fill)=>`<path d="M${p[0]} ${p[1]-10} l10 10 -10 10 -10 -10 z" fill="${fill}" class="site"/>`;
  function network({snaps=false,route=false,sites=[],absolute=null,weights=m.weights,assign=false,title='Worked network'}={}){
    let body='';
    m.edges.forEach(([u,v])=>{body+=line(m.nodes[u],m.nodes[v],'road')});
    if(route){const path=[0,9,1,2,3,6,7];for(let i=1;i<path.length;i++)body+=line(m.nodes[path[i-1]],m.nodes[path[i]],'route');body+=line(m.nodes[0],m.nodes[7],'shortcut');}
    if(snaps)m.raw.forEach((p,i)=>{body+=line(p,m.nodes[m.demand[i]],'connector');body+=`<circle cx="${p[0]}" cy="${p[1]}" r="4.5" class="raw"/>`;});
    m.nodes.forEach(([x,y],i)=>{body+=`<circle cx="${x}" cy="${y}" r="3.5" class="node"/>`;body+=text(x,y-13,m.names[i],'node-label','text-anchor="middle"');});
    const allocation=sites.length&&assign?m.assignments(sites):[];
    m.demand.forEach((q,i)=>{const [x,y]=m.nodes[q],c=allocation.length?colors[allocation[i]]:colors[0];body+=`<circle cx="${x}" cy="${y}" r="${7+Math.min(weights[i],6)*.7}" fill="${c}" class="demand"/>`;body+=text(x,y+24,`w=${weights[i]}`,'weight-label','text-anchor="middle"');});
    sites.forEach((q,i)=>{body+=diamond(m.nodes[q],colors[i%colors.length]);});
    if(absolute){const p=m.location(absolute);body+=diamond(p,'#b74991');body+=text(p[0],p[1]-19,'absolute','annotation','text-anchor="middle"');}
    if(snaps)body+=text(80,56,'J: inserted','annotation','text-anchor="middle"');
    if(route)body+=text(287,183,'Euclidean shortcut','annotation','text-anchor="middle"');
    return wrap(title,body);
  }
  function bars(values,labels,{best=-1,ytitle='Weighted distance (weight·m)',title='Objective by candidate'}={}){
    const max=Math.max(...values,1),left=54,width=490,top=22,height=150,step=width/values.length;
    let b='';for(let i=0;i<=3;i++){const v=max*i/3,y=top+height-height*i/3;b+=line([left,y],[left+width,y],'gridline');b+=text(left-8,y+4,Math.round(v),'axis-label','text-anchor="end"');}
    values.forEach((v,i)=>{const x=left+i*step+step*.18,y=top+height-height*v/max;const fill=i===best?'#2478bd':'var(--bar)';b+=`<rect x="${x}" y="${y}" width="${step*.64}" height="${height*v/max}" rx="3" fill="${fill}"/>`;b+=text(x+step*.32,y-5,num(v),'axis-label','text-anchor="middle"');b+=text(x+step*.32,top+height+19,labels[i],'axis-label','text-anchor="middle"');});
    b+=text(left+width/2,218,ytitle,'annotation','text-anchor="middle"');return wrap(title,b,'0 0 570 235');
  }
  function envelope(withOffsets=false){
    const edge=3,[u,v]=m.edges[edge],L=m.length(u,v),max=520,left=55,top=25,width=480,height=190;
    const x=t=>left+t/L*width,y=d=>top+height-d/max*height;
    const f=(q,i,t)=>Math.min(m.D[q][u]+t,m.D[q][v]+L-t)+(withOffsets?m.offsets[i]:0);
    let b='';for(let i=0;i<=4;i++){const val=max*i/4;b+=line([left,y(val)],[left+width,y(val)],'gridline');b+=text(left-9,y(val)+4,Math.round(val),'axis-label','text-anchor="end"');}
    const path=fn=>Array.from({length:131},(_,i)=>`${i?'L':'M'}${num(x(L*i/130))},${num(y(fn(L*i/130)))}`).join(' ');
    m.demand.forEach((q,i)=>{b+=`<path d="${path(t=>f(q,i,t))}" fill="none" stroke="var(--tent)" stroke-width="1.5"/>`;});
    const env=t=>Math.max(...m.demand.map((q,i)=>f(q,i,t)));b+=`<path d="${path(env)}" class="envelope"/>`;
    const a=m.absolute(withOffsets);if(a.edge===edge){b+=line([x(a.t),y(a.radius)],[x(a.t),top+height],'connector');b+=`<circle cx="${x(a.t)}" cy="${y(a.radius)}" r="5" fill="#b74991"/>`;b+=text(x(a.t)+9,y(a.radius)-12,`${num(a.radius)} m`,'annotation');}
    b+=text(left,top+height+20,'C · 0 m','axis-label');b+=text(left+width,top+height+20,`D · ${L} m`,'axis-label','text-anchor="end"');b+=text(left+width/2,260,'Position t along C–D','annotation','text-anchor="middle"');return wrap('Individual distance tents and their maximum along C to D',b,'0 0 570 275');
  }
  const api={network,bars,envelope};if(typeof module==='object'&&module.exports)module.exports=api;else root.NetCenterVisuals=api;
})(typeof globalThis==='object'?globalThis:this);
