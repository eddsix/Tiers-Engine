(() => {
"use strict";

/*
 v4.3: stability-first rebuild.
 There is intentionally NO recursive "model(history prefix)" call inside the
 live render. That was the source of the freeze pattern in previous builds.
 The live model calculates every signal from one bounded current history.
 Walk-forward validation is calculated incrementally from recorded predictions.
*/

const TIERS=new Set([27,13,36,11,30,8,23,10,5,24,16,33]);
const WHEEL=[0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];
const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const KEY="tiers-engine-v43-state";
const OLD=["tiers-engine-v42-state","tiers-engine-v4-state","tiers-engine-v3-history"];
const EXPERTS=["recent","transition","interval","streak","jump","avgJump3","direction","position","sequence","alternation"];
const MIN=12, MAX=500;

let S={version:"4.3",history:[],predictions:[]};
let busy=false;

const $=id=>document.getElementById(id);
const tier=n=>TIERS.has(n);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const pct=p=>(100*p).toFixed(1)+"%";
const safe=a=>Array.isArray(a)?a.map(Number).filter(n=>Number.isInteger(n)&&n>=0&&n<=36):[];

function load(){
  try{
    const raw=localStorage.getItem(KEY);
    if(raw){
      const x=JSON.parse(raw);
      if(x&&Array.isArray(x.history)){S={version:"4.3",history:safe(x.history),predictions:Array.isArray(x.predictions)?x.predictions:[]};return;}
    }
    for(const k of OLD){
      const rawOld=localStorage.getItem(k); if(!rawOld)continue;
      const x=JSON.parse(rawOld);
      const h=Array.isArray(x)?x:(x&&Array.isArray(x.history)?x.history:[]);
      if(h.length){S.history=safe(h);break;}
    }
  }catch(e){console.error(e);S={version:"4.3",history:[],predictions:[]}}
}
function save(){
  try{localStorage.setItem(KEY,JSON.stringify(S));return true}
  catch(e){console.error(e);return false}
}
function wi(n){return WHEEL.indexOf(n)}
function dist(a,b){const x=wi(a),y=wi(b);if(x<0||y<0)return 0;const d=Math.abs(x-y);return Math.min(d,37-d)}
function dir(a,b){const x=wi(a),y=wi(b);if(x<0||y<0)return 0;const cw=(y-x+37)%37,ccw=(x-y+37)%37;return cw===ccw?0:(cw<ccw?1:-1)}
function beta(s,n){return (s+2)/(n+4)}
function rate(a){return a.length?a.filter(tier).length/a.length:.5}
function interval(c,k){
  let n=0;for(let i=k;i>=0;i--){if(tier(c[i]))break;n++}return n
}
function streak(c,k){
  if(k<0)return {kind:null,len:0};const kind=tier(c[k]);let n=0;
  for(let i=k;i>=0&&tier(c[i])===kind;i--)n++;return {kind,len:n}
}
function pos(n){const i=wi(n);return i<0?0:Math.floor(i/5)}
function jumpB(j){return Math.min(6,Math.floor(j/3))}
function avg3(c,k){
  if(k<1)return null;const ds=[];for(let i=Math.max(1,k-2);i<=k;i++)ds.push(dist(c[i-1],c[i]));
  return ds.reduce((a,b)=>a+b,0)/ds.length
}
function seq(c,k,len){if(k<len-1)return "";return c.slice(k-len+1,k+1).map(tier?"":"")}
function bits(c,k,len){
  if(k<len-1)return null;let s="";for(let i=k-len+1;i<=k;i++)s+=tier(c[i])?"1":"0";return s
}
function alt(c,k){
  const b=bits(c,k,4);return b?b[0]!==b[1]&&b[1]!==b[2]&&b[2]!==b[3]:false
}
function conditional(c,match){
  let s=0,n=0;for(let k=1;k<c.length;k++){if(match(k)){n++;if(tier(c[k]))s++}}
  return {p:beta(s,n),n}
}

/* Current evidence only. No nested walk-forward recalculation. */
function experts(h){
  const c=h.length>MAX?h.slice(-MAX):h;
  if(c.length<MIN)return {};
  const o={};

  const ws=[1,1.2,1.5,1.9,2.3,2.8], wins=[5,10,20,50,100,250];
  let z=0,w=0;for(let i=0;i<wins.length;i++){z+=ws[i]*rate(c.slice(Math.max(0,c.length-wins[i])));w+=ws[i]}
  o.recent={p:z/w,n:c.length};

  const lt=tier(c[c.length-1]);
  o.transition=conditional(c,k=>tier(c[k-1])===lt);

  const ci=interval(c,c.length-1);
  o.interval=conditional(c,k=>interval(c,k-1)===ci);

  const sb=streak(c,c.length-1);
  o.streak=conditional(c,k=>{const q=streak(c,k-1);return q.kind===sb.kind&&Math.min(q.len,8)===Math.min(sb.len,8)});

  const cj=c.length>1?dist(c[c.length-2],c[c.length-1]):0;
  o.jump=conditional(c,k=>k>1&&jumpB(dist(c[k-2],c[k-1]))===jumpB(cj));

  const ca=avg3(c,c.length-1);
  o.avgJump3=conditional(c,k=>{if(k<2||ca==null)return false;const a=avg3(c,k-1);return a!=null&&Math.abs(a-ca)<3});

  const cd=c.length>1?dir(c[c.length-2],c[c.length-1]):0;
  o.direction=conditional(c,k=>k>1&&dir(c[k-2],c[k-1])===cd);

  const cp=pos(c[c.length-1]);
  o.position=conditional(c,k=>pos(c[k-1])===cp);

  const cs=bits(c,c.length-1,3);
  o.sequence=conditional(c,k=>k>=3&&bits(c,k-1,3)===cs);

  const altNow=alt(c,c.length-1);
  o.alternation=conditional(c,k=>k>=4&&alt(c,k-1)===altNow);
  return o;
}

/* Adaptive weights are based on historical recorded prediction errors.
   They are updated once per completed prediction, never recursively on render. */
function learnedWeights(){
  const w=Object.fromEntries(EXPERTS.map(x=>[x,1]));
  const rows=S.predictions.filter(x=>x&&x.experts&&x.actual!=null).slice(-250);
  for(const r of rows){
    for(const f of EXPERTS){
      const q=r.experts[f];if(!q||q.n<3)continue;
      const e=q.p-(tier(Number(r.actual))?1:0);
      w[f]=clamp(w[f]*Math.exp(-.08*e*e),.2,5);
    }
  }
  return w;
}
function combine(ps,w){
  let s=0,z=0;
  for(const f of EXPERTS){const q=ps[f];if(!q||q.n<1)continue;const ww=w[f]||1;s+=ww*q.p;z+=ww}
  return z?s/z:.5;
}
function current(h){
  if(h.length<MIN)return {ready:false};
  const ps=experts(h),w=learnedWeights(),p=combine(ps,w);
  const active=EXPERTS.filter(f=>ps[f]&&ps[f].n>=3).length;
  const edge=Math.abs(p-.5);
  let conf="Baja";
  if(h.length>=80&&edge>=.12&&active>=6)conf="Alta";
  else if(h.length>=35&&edge>=.06&&active>=4)conf="Media";
  return {ready:true,p,ps,w,active,edge,conf};
}
function validation(){
  const rows=S.predictions.filter(x=>x&&x.prediction&&x.actual!=null);
  if(!rows.length)return {n:0};
  let b=0,base=0,cal=0;
  for(const r of rows.slice(-250)){
    const y=tier(Number(r.actual))?1:0,p=clamp(Number(r.prediction),.001,.999);
    b+=(p-y)*(p-y);
  }
  const rr=rows.slice(-250);
  for(let i=0;i<rr.length;i++){
    const past=rr.slice(0,i),y=tier(Number(rr[i].actual))?1:0;
    const bp=past.length?rate(past.map(x=>Number(x.actual))):.5;
    base+=(bp-y)*(bp-y);
  }
  const n=rr.length;return {n,brier:b/n,base:b?base/n:null};
}
function render(){
  try{
    const h=S.history,m=current(h),v=validation();
    $("counter").textContent=h.length+" spins";$("spins").textContent=h.length;
    $("history").innerHTML=h.slice(0,50).map(n=>`<span class="result ${n===0?"green":RED.has(n)?"red":"black"}">${n}</span>`).join("")||"—";
    if(!m.ready){
      $("tiersProb").textContent="—";$("noTiersProb").textContent="—";$("mainSide").textContent="—";
      $("confidence").textContent=h.length?`Datos insuficientes (${h.length}/${MIN})`:"—";$("evidence").textContent="Insuficiente";
      $("modelEdge").textContent="—";$("validationScore").textContent="—";$("signal").textContent="ESPERANDO DATOS";$("signal").className="signal";
      $("modelStatus").innerHTML=`<span class="warn">Los resultados se guardan desde el primer spin. Las señales condicionales se activan con ${MIN} spins.</span>`;
      $("signals").innerHTML="—";$("tiersBar").style.width="0";$("noTiersBar").style.width="0";
      ["validationCount","brier","baselineBrier","improvement","calibration"].forEach(id=>$(id).textContent="—");return;
    }
    const p=m.p;
    $("tiersProb").textContent=pct(p);$("noTiersProb").textContent=pct(1-p);
    $("tiersBar").style.width=(100*p)+"%";$("noTiersBar").style.width=(100*(1-p))+"%";
    $("mainSide").textContent=p>=.5?"Tiers · "+pct(p):"No Tiers · "+pct(1-p);
    $("confidence").textContent=m.conf;$("evidence").textContent=m.active+"/10";$("modelEdge").textContent=(m.edge*100).toFixed(1)+" pp";
    $("validationScore").textContent=v.n?(Math.max(0,1-v.brier/.25)*100).toFixed(1)+"%":"—";
    $("signal").textContent=m.conf==="Alta"?"SEÑAL FUERTE":m.conf==="Media"?"SEÑAL MODERADA":"SEÑAL DÉBIL";
    $("signal").className="signal "+(m.conf==="Alta"?"high":m.conf==="Media"?"medium":"");
    $("modelStatus").innerHTML=`<div><b>Estimación:</b> probabilidad modelada de Tiers en la próxima tirada.</div>
      <div><b>Motor:</b> ensemble adaptativo de 10 señales, con pesos aprendidos a partir de errores históricos registrados.</div>
      <div><b>Protección:</b> la predicción actual no ejecuta un walk-forward recursivo; un fallo de una señal no puede bloquear la entrada.</div>`;
    const names={recent:"Frecuencia reciente",transition:"Transiciones",interval:"Intervalos",streak:"Rachas",jump:"Magnitud de salto",avgJump3:"Media últimos 3 saltos",direction:"Dirección física",position:"Posición cilindro",sequence:"Secuencias",alternation:"Alternancia"};
    $("signals").innerHTML=EXPERTS.map(f=>{const q=m.ps[f];if(!q)return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>—</span></div><small>Sin evidencia</small></div>`;const d=q.p-.5;return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>${pct(q.p)}</span></div><small>${q.n} casos · ${d>=0?"+":""}${(d*100).toFixed(1)} pp · peso ${m.w[f].toFixed(2)}×</small></div>`}).join("");
    $("validationCount").textContent=v.n+" predicciones";$("brier").textContent=v.n?v.brier.toFixed(4):"—";$("baselineBrier").textContent=v.n&&v.base!=null?v.base.toFixed(4):"—";
    $("improvement").textContent=v.n&&v.base?vct(v.base,v.brier):"—";$("calibration").textContent="—";
  }catch(e){
    console.error("render",e);
    $("modelStatus").innerHTML='<span class="warn">Error de análisis aislado. El histórico permanece guardado y la entrada sigue disponible.</span>';
  }
}
function vct(base,b){return pct(clamp((base-b)/Math.max(.0001,base),-1,1))}
function add(n){
  if(busy)return;busy=true;
  try{
    const before=S.history.slice(), m=current(before);
    S.history.unshift(Number(n));
    S.predictions.unshift({
      spin:before.length+1,
      prediction:m.ready?m.p:null,
      actual:Number(n),
      experts:m.ready?m.ps:null,
      at:new Date().toISOString()
    });
    // Persistence occurs before any optional UI/model work.
    save();render();
  }catch(e){console.error("add",e);save();render()}
  finally{busy=false}
}
function undo(){if(busy||!S.history.length)return;S.history.shift();if(S.predictions.length)S.predictions.shift();save();render()}
function clearAll(){if(busy)return;if(confirm("¿Borrar todo el histórico?")){S={version:"4.3",history:[],predictions:[]};save();render()}}
$("theme").onclick=()=>{document.body.classList.toggle("night");const n=document.body.classList.contains("night");localStorage.setItem("tiers-engine-theme",n?"night":"day");$("theme").textContent=n?"Modo día":"Modo noche"};
$("undo").onclick=undo;$("clear").onclick=clearAll;
for(let n=0;n<=36;n++){const b=document.createElement("button");b.className="num "+(n===0?"green":RED.has(n)?"red":"black");b.textContent=n;b.type="button";b.onclick=()=>add(n);$("numbers").appendChild(b)}
load();if(localStorage.getItem("tiers-engine-theme")==="night"){document.body.classList.add("night");$("theme").textContent="Modo día"}render();
})();