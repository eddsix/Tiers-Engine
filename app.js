(() => {
"use strict";

/*
 Tiers Engine v4.2
 Critical v4.1 bug fixed:
 - sequence/alternation mapping now uses real callbacks instead of map(1),
   which previously threw a TypeError exactly when the conditional experts
   became active around the 18-spin threshold.
 Performance:
 - walk-forward learning uses bounded data and compact feature scans.
 - render is protected so one model error cannot destroy the input UI.
*/

const TIERS = new Set([27,13,36,11,30,8,23,10,5,24,16,33]);
const WHEEL = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];
const RED = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

const STORE = "tiers-engine-v42-state";
const OLD_STORES = ["tiers-engine-v4-state","tiers-engine-v3-history"];
const MIN_TRAIN = 12;
const LEARN_START = 18;
const MAX_ANALYSIS = 300;
const MAX_LEARN = 180;
const EXPERTS = ["recent","transition","interval","streak","jump","avgJump3","direction","position","sequence","alternation"];

let state = {version:"4.2",history:[],predictions:[]};
let busy = false;

const $ = id => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const pct = p => `${(100*p).toFixed(1)}%`;
const isTiers = n => TIERS.has(n);
const logit = p => Math.log(clamp(p,.001,.999)/(1-clamp(p,.001,.999)));
const sigmoid = z => 1/(1+Math.exp(-clamp(z,-20,20)));

function safeHistory(h){
  return Array.isArray(h) ? h.map(Number).filter(n=>Number.isInteger(n)&&n>=0&&n<=36) : [];
}
function load(){
  try{
    const raw=localStorage.getItem(STORE);
    if(raw){
      const x=JSON.parse(raw);
      if(x && Array.isArray(x.history)){
        state={version:"4.2",history:safeHistory(x.history),predictions:Array.isArray(x.predictions)?x.predictions:[]};
        return;
      }
    }
    for(const key of OLD_STORES){
      const rawOld=localStorage.getItem(key);
      if(!rawOld) continue;
      const x=JSON.parse(rawOld);
      const h=Array.isArray(x)?x:(x&&Array.isArray(x.history)?x.history:[]);
      if(h.length){state.history=safeHistory(h);break;}
    }
  }catch(e){
    console.error("load",e);
    state={version:"4.2",history:[],predictions:[]};
  }
}
function save(){
  try{
    localStorage.setItem(STORE,JSON.stringify(state));
    return true;
  }catch(e){
    console.error("save",e);
    return false;
  }
}
function wheelIndex(n){return WHEEL.indexOf(n)}
function distance(a,b){
  const ia=wheelIndex(a),ib=wheelIndex(b);
  if(ia<0||ib<0)return 0;
  const d=Math.abs(ia-ib);
  return Math.min(d,37-d);
}
function direction(a,b){
  const ia=wheelIndex(a),ib=wheelIndex(b);
  if(ia<0||ib<0)return 0;
  const cw=(ib-ia+37)%37,ccw=(ia-ib+37)%37;
  return cw===ccw?0:(cw<ccw?1:-1);
}
function beta(s,n,prior=2){return (s+prior)/(n+2*prior)}
function rate(a){return a.length?a.filter(isTiers).length/a.length:.5}
function streakAt(c,k){
  if(k<0)return {kind:null,len:0};
  const kind=isTiers(c[k]); let len=0;
  for(let j=k;j>=0&&isTiers(c[j])===kind;j--)len++;
  return {kind,len};
}
function intervalAt(c,k){
  let n=0;
  for(let j=k;j>=0;j--){ if(isTiers(c[j])) break; n++; }
  return n;
}
function avgJumpAt(c,k){
  if(k<1)return null;
  const start=Math.max(1,k-2), ds=[];
  for(let i=start;i<=k;i++) ds.push(distance(c[i-1],c[i]));
  return ds.length?ds.reduce((a,b)=>a+b,0)/ds.length:null;
}
function posBucket(n){
  const i=wheelIndex(n);
  return i<0?0:Math.floor(i/5);
}
function jumpBucket(j){
  // Data-defined 7 broad bins over the 0..18 circular-distance range.
  return Math.min(6,Math.floor(j/3));
}
function bits(c,end,len){
  const s=c.slice(Math.max(0,end-len+1),end+1);
  return s.map(x=>isTiers(x)?1:0).join("");
}
function alt4(c,end){
  if(end<3)return false;
  const q=bits(c,end,4);
  return q[0]!==q[1]&&q[1]!==q[2]&&q[2]!==q[3];
}

function conditional(c,predicate){
  let s=0,n=0;
  for(let k=1;k<c.length;k++){
    if(predicate(k,c)){n++;if(isTiers(c[k]))s++}
  }
  return {p:beta(s,n),n};
}

/* One compact pass over a bounded context. */
function expertProbs(history){
  const a=safeHistory(history);
  if(a.length<MIN_TRAIN)return {};
  const c=a.length>MAX_ANALYSIS?a.slice(-MAX_ANALYSIS):a;
  const out={};

  const weights=[1,1.25,1.6,2,2.5,3];
  let num=0,den=0;
  for(let i=0;i<weights.length;i++){
    const w=weights[i], start=Math.max(0,c.length-[5,10,20,50,100,250][i]);
    num+=w*rate(c.slice(start)); den+=w;
  }
  out.recent={p:num/den,n:c.length};

  const lastT=isTiers(c[c.length-1]);
  out.transition=conditional(c,(k,x)=>isTiers(x[k-1])===lastT);

  const curI=intervalAt(c,c.length-1);
  out.interval=conditional(c,(k,x)=>intervalAt(x,k-1)===curI);

  const sb=streakAt(c,c.length-1);
  out.streak=conditional(c,(k,x)=>{
    const z=streakAt(x,k-1);
    return z.kind===sb.kind && Math.min(z.len,8)===Math.min(sb.len,8);
  });

  const curJ=c.length>=2?distance(c[c.length-2],c[c.length-1]):0;
  out.jump=conditional(c,(k,x)=>k>=2&&jumpBucket(distance(x[k-2],x[k-1]))===jumpBucket(curJ));

  const curA=avgJumpAt(c,c.length-1);
  out.avgJump3=conditional(c,(k,x)=>{
    if(k<2||curA==null)return false;
    const av=avgJumpAt(x,k-1);
    return av!=null&&Math.abs(av-curA)<3;
  });

  const curD=c.length>=2?direction(c[c.length-2],c[c.length-1]):0;
  out.direction=conditional(c,(k,x)=>k>=2&&direction(x[k-2],x[k-1])===curD);

  const curPos=posBucket(c[c.length-1]);
  out.position=conditional(c,(k,x)=>posBucket(x[k-1])===curPos);

  const sk=bits(c,c.length-1,3);
  out.sequence=conditional(c,(k,x)=>k>=3&&bits(x,k-1,3)===sk);

  const curAlt=alt4(c,c.length-1);
  out.alternation=conditional(c,(k,x)=>k>=4&&alt4(x,k-1)===curAlt);

  return out;
}

function combine(probs,weights){
  let z=0,wSum=0;
  for(const f of EXPERTS){
    const q=probs[f];
    if(!q||q.n<1)continue;
    const w=Math.max(.05,weights[f]||1);
    z+=w*logit(q.p); wSum+=w;
  }
  return wSum?sigmoid(z/wSum):.5;
}

function learn(history){
  const a=safeHistory(history);
  const c=a.length>MAX_LEARN?a.slice(-MAX_LEARN):a;
  const w=Object.fromEntries(EXPERTS.map(f=>[f,1]));
  const preds=[];
  if(c.length<LEARN_START)return {weights:w,preds,brier:null,baseBrier:null,calibration:null};

  const eta=.08;
  for(let k=LEARN_START;k<c.length;k++){
    const past=c.slice(0,k);
    const ps=expertProbs(past);
    const p=combine(ps,w);
    const y=isTiers(c[k])?1:0;
    preds.push({p,y});
    for(const f of EXPERTS){
      const q=ps[f];
      if(!q||q.n<3)continue;
      const e=q.p-y;
      // Brier-style adaptive update: experts that historically track the
      // observed side better gain weight; poor experts lose weight.
      w[f]=clamp(w[f]*Math.exp(-eta*e*e),.15,8);
    }
  }
  if(!preds.length)return {weights:w,preds,brier:null,baseBrier:null,calibration:null};

  const brier=preds.reduce((s,q)=>s+(q.p-q.y)**2,0)/preds.length;
  let base=0;
  for(let k=LEARN_START;k<c.length;k++){
    const past=c.slice(0,k), p=beta(past.filter(isTiers).length,past.length);
    base+=(p-(isTiers(c[k])?1:0))**2;
  }
  base/=preds.length;

  const bins=Array.from({length:5},()=>({n:0,p:0,y:0}));
  for(const q of preds){
    const i=Math.min(4,Math.floor(q.p*5));
    bins[i].n++;bins[i].p+=q.p;bins[i].y+=q.y;
  }
  let cal=0,ncal=0;
  for(const b of bins)if(b.n){
    cal+=Math.abs(b.p/b.n-b.y/b.n)*b.n;ncal+=b.n;
  }
  return {weights:w,preds,brier,baseBrier:base,calibration:ncal?1-cal/ncal:null};
}

function currentModel(history){
  const a=safeHistory(history);
  if(a.length<MIN_TRAIN)return {ready:false,p:.5,weights:null,probs:{}};
  const learned=learn(a);
  const ps=expertProbs(a);
  const p=combine(ps,learned.weights);
  const active=EXPERTS.filter(f=>ps[f]&&ps[f].n>=3).length;
  const edge=Math.abs(p-.5);
  const validation=learned.brier==null?0:clamp(1-learned.brier/.25,0,1);
  let confidence="Baja";
  if(a.length>=80&&edge>=.12&&validation>=.10&&active>=6)confidence="Alta";
  else if(a.length>=35&&edge>=.06&&validation>=.03&&active>=4)confidence="Media";
  return {ready:true,p,weights:learned.weights,probs:ps,confidence,active,validation,edge,learned};
}

function render(){
  try{
    const a=state.history;
    const m=currentModel(a);
    $("counter").textContent=`${a.length} spins`;
    $("spins").textContent=a.length;
    $("history").innerHTML=a.slice(0,50).map(n=>`<span class="result ${n===0?"green":RED.has(n)?"red":"black"}">${n}</span>`).join("")||"—";

    if(!m.ready){
      $("tiersProb").textContent="—";$("noTiersProb").textContent="—";$("mainSide").textContent="—";
      $("confidence").textContent=a.length?`Datos insuficientes (${a.length}/${MIN_TRAIN})`:"—";
      $("evidence").textContent="Insuficiente";$("modelEdge").textContent="—";$("validationScore").textContent="—";
      $("signal").textContent=a.length<MIN_TRAIN?"ESPERANDO DATOS":"—";$("signal").className="signal";
      $("modelStatus").innerHTML=`<span class="warn">Se necesitan ${MIN_TRAIN} spins para activar las señales condicionales. Los datos ya quedan guardados desde el primer spin.</span>`;
      $("signals").innerHTML="—";
      ["validationCount","brier","baselineBrier","improvement","calibration"].forEach(id=>$(id).textContent="—");
      $("tiersBar").style.width="0";$("noTiersBar").style.width="0";
      return;
    }

    const p=m.p,np=1-p;
    $("tiersProb").textContent=pct(p);$("noTiersProb").textContent=pct(np);
    $("tiersBar").style.width=`${100*p}%`;$("noTiersBar").style.width=`${100*np}%`;
    $("mainSide").textContent=p>=.5?`Tiers · ${pct(p)}`:`No Tiers · ${pct(np)}`;
    $("confidence").textContent=m.confidence;
    $("evidence").textContent=`${m.active}/${EXPERTS.length} señales`;
    $("modelEdge").textContent=`${(m.edge*100).toFixed(1)} pp`;
    $("validationScore").textContent=m.learned.brier==null?"—":pct(m.validation);
    $("signal").textContent=m.confidence==="Alta"?"SEÑAL FUERTE":m.confidence==="Media"?"SEÑAL MODERADA":"SEÑAL DÉBIL";
    $("signal").className="signal "+(m.confidence==="Alta"?"high":m.confidence==="Media"?"medium":"");

    $("modelStatus").innerHTML=
      `<div><b>Estimación:</b> probabilidad modelada de Tiers en la próxima tirada.</div>
       <div><b>Motor:</b> ensemble adaptativo; los pesos se actualizan con rendimiento histórico walk-forward.</div>
       <div><b>Datos:</b> ventanas 5 / 10 / 20 / 50 / 100 / 250 / histórico completo cuando hay muestra disponible.</div>
       <div><b>Protección:</b> cálculo acotado y manejo de errores para que una señal defectuosa no bloquee la entrada ni borre el historial.</div>`;

    const names={
      recent:"Frecuencia reciente",transition:"Transiciones",interval:"Intervalos",
      streak:"Rachas",jump:"Magnitud de salto",avgJump3:"Media últimos 3 saltos",
      direction:"Dirección física",position:"Posición cilindro",sequence:"Secuencias",alternation:"Alternancia"
    };
    $("signals").innerHTML=EXPERTS.map(f=>{
      const q=m.probs[f],w=m.weights[f];
      if(!q)return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>—</span></div><small>Sin evidencia suficiente</small></div>`;
      const d=q.p-.5;
      return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>${pct(q.p)}</span></div><small>${q.n} casos · ${d>=0?"+":""}${(d*100).toFixed(1)} pp · peso <span class="weight">${w.toFixed(2)}×</span></small></div>`;
    }).join("");

    const l=m.learned;
    $("validationCount").textContent=`${l.preds.length} predicciones`;
    $("brier").textContent=l.brier==null?"—":l.brier.toFixed(4);
    $("baselineBrier").textContent=l.baseBrier==null?"—":l.baseBrier.toFixed(4);
    $("improvement").textContent=l.brier==null?"—":pct(clamp((l.baseBrier-l.brier)/Math.max(.0001,l.baseBrier),-1,1));
    $("calibration").textContent=l.calibration==null?"—":pct(clamp(l.calibration,0,1));
  }catch(e){
    console.error("render",e);
    $("modelStatus").innerHTML=`<span class="warn">El histórico está protegido. El motor ha detenido el cálculo de la señal que produjo un error para mantener la aplicación operativa.</span>`;
  }
}

function add(n){
  if(busy)return;
  busy=true;
  try{
    const before=state.history.slice();
    let pred=null;
    try{
      const m=currentModel(before);
      if(m.ready)pred={p:m.p,confidence:m.confidence};
    }catch(e){console.error("prediction before add",e)}
    state.predictions.unshift({spin:before.length+1,prediction:pred,actual:n,at:new Date().toISOString()});
    state.history.unshift(n);
    // Save BEFORE rendering. Even if rendering/modeling fails, the spin survives refresh.
    save();
    render();
  }finally{
    busy=false;
  }
}
function undo(){
  if(busy||!state.history.length)return;
  state.history.shift();
  if(state.predictions.length)state.predictions.shift();
  save();render();
}
function clearAll(){
  if(busy)return;
  if(confirm("¿Borrar todo el histórico?")){
    state={version:"4.2",history:[],predictions:[]};
    save();render();
  }
}

$("theme").onclick=()=>{
  document.body.classList.toggle("night");
  const night=document.body.classList.contains("night");
  localStorage.setItem("tiers-engine-theme",night?"night":"day");
  $("theme").textContent=night?"Modo día":"Modo noche";
};
$("undo").onclick=undo;
$("clear").onclick=clearAll;

for(let n=0;n<=36;n++){
  const b=document.createElement("button");
  b.className=`num ${n===0?"green":RED.has(n)?"red":"black"}`;
  b.textContent=n;b.type="button";
  b.onclick=()=>add(n);
  $("numbers").appendChild(b);
}

load();
if(localStorage.getItem("tiers-engine-theme")==="night"){
  document.body.classList.add("night");
  $("theme").textContent="Modo día";
}
render();
})();
