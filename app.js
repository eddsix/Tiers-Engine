(() => {
"use strict";

/* Tiers Engine v4.0
   - Final prediction is Tiers vs No Tiers.
   - All historical evaluation is walk-forward: no future information is used.
   - Signal weights are learned from out-of-sample historical loss.
*/

const TIERS = new Set([27,13,36,11,30,8,23,10,5,24,16,33]);
const WHEEL = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];
const RED = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const STORE = "tiers-engine-v4-state";
const OLD_STORE = "tiers-engine-v3-history";
const MIN_TRAIN = 18;
const EXPERTS = ["recent","transition","interval","streak","jump","avgJump3","direction","position","sequence","alternation"];

let state = {version:"4.0", history:[], predictions:[]};

const $ = id => document.getElementById(id);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const isTiers = n => TIERS.has(n);
const pct = p => `${(100*p).toFixed(1)}%`;
const logit = p => Math.log(clamp(p,.001,.999)/(1-clamp(p,.001,.999)));
const sigmoid = z => 1/(1+Math.exp(-clamp(z,-20,20)));

function load(){
  try{
    const raw=localStorage.getItem(STORE);
    if(raw){
      const x=JSON.parse(raw);
      if(x && Array.isArray(x.history)) state={version:"4.0",history:x.history.filter(n=>Number.isInteger(n)&&n>=0&&n<=36),predictions:Array.isArray(x.predictions)?x.predictions:[]};
    }else{
      const old=localStorage.getItem(OLD_STORE);
      if(old){
        const h=JSON.parse(old);
        if(Array.isArray(h)) state.history=h.filter(n=>Number.isInteger(n)&&n>=0&&n<=36);
      }
    }
  }catch(e){console.error(e)}
}
function save(){localStorage.setItem(STORE,JSON.stringify(state))}
function rate(a){return a.length?a.filter(isTiers).length/a.length:.5}
function wheelIndex(n){return WHEEL.indexOf(n)}
function distance(a,b){const ia=wheelIndex(a),ib=wheelIndex(b);if(ia<0||ib<0)return 0;const d=Math.abs(ia-ib);return Math.min(d,WHEEL.length-d)}
function direction(a,b){
  const ia=wheelIndex(a),ib=wheelIndex(b);if(ia<0||ib<0)return 0;
  const cw=(ib-ia+37)%37,ccw=(ia-ib+37)%37;
  return cw===ccw?0:(cw<ccw?1:-1);
}
function intervalsChron(c){
  const out=new Array(c.length).fill(0), run=0;
  for(let i=0;i<c.length;i++){
    out[i]=run;
    if(isTiers(c[i])) run=0; else run++;
  }
  return out;
}
function streakBefore(c){
  if(!c.length)return {kind:null,len:0};
  const last=isTiers(c[c.length-1]); let n=0;
  for(let i=c.length-1;i>=0&&isTiers(c[i])===last;i--)n++;
  return {kind:last,len:n};
}
function jumpBucket(j){return Math.min(6,Math.floor(j/4))}
function avg3(c){
  if(c.length<2)return null;
  const ds=[];for(let i=Math.max(1,c.length-3);i<c.length;i++)ds.push(distance(c[i-1],c[i]));
  return ds.reduce((a,b)=>a+b,0)/ds.length;
}
function positionBucket(n){const i=wheelIndex(n);return i<0?0:Math.floor(i/6)}
function seqKey(c,len){
  if(c.length<len)return null;
  return c.slice(-len).map(isTiers?1:0).join("");
}
function beta(success,total,prior=2){
  return (success+prior)/(total+2*prior);
}
function conditional(c, predicate){
  let s=0,n=0;
  for(let k=1;k<c.length;k++){
    if(predicate(k,c)){n++;if(isTiers(c[k]))s++}
  }
  return {p:beta(s,n),n};
}

/* Each expert produces its own probability from past data only. */
function expertProbs(a){
  if(a.length<MIN_TRAIN)return {};
  const c=a.slice().reverse(), out={};
  const wins=[5,10,20,50,100,250].map(w=>rate(c.slice(Math.max(0,c.length-w))));
  out.recent={p:wins.reduce((s,p,i)=>s+p/[1,1.3,1.6,2.0,2.5,3][i],0)/wins.reduce((s,_,i)=>s+1/[1,1.3,1.6,2.0,2.5,3][i],0),n:Math.min(c.length,250)};

  const lastT=isTiers(c[c.length-1]);
  out.transition=conditional(c,(k,x)=>isTiers(x[k-1])===lastT);

  const ints=intervalsChron(c), curI=ints[c.length-1];
  out.interval=conditional(c,(k,x)=>ints[k]===curI);
  const sb=streakBefore(c);
  out.streak=conditional(c,(k,x)=>{
    const s=streakBefore(x.slice(0,k));
    return s.kind===sb.kind && Math.min(s.len,8)===Math.min(sb.len,8);
  });

  const curJ=c.length>=2?distance(c[c.length-2],c[c.length-1]):0;
  out.jump=conditional(c,(k,x)=>k>=2 && jumpBucket(distance(x[k-2],x[k-1]))===jumpBucket(curJ));

  const curA=avg3(c);
  out.avgJump3=conditional(c,(k,x)=>{
    if(k<2||curA==null)return false;
    const av=avg3(x.slice(0,k));
    return av!=null && Math.abs(av-curA)<2.5;
  });

  const curD=c.length>=2?direction(c[c.length-2],c[c.length-1]):0;
  out.direction=conditional(c,(k,x)=>k>=2 && direction(x[k-2],x[k-1])===curD);

  const curPos=positionBucket(c[c.length-1]);
  out.position=conditional(c,(k,x)=>positionBucket(x[k-1])===curPos);

  const sk=seqKey(c,3);
  out.sequence=conditional(c,(k,x)=>k>=3 && seqKey(x.slice(0,k),3)===sk);

  const bits=c.slice(-4).map(isTiers?1:0).join("");
  out.alternation=conditional(c,(k,x)=>{
    if(k<4)return false;
    const q=x.slice(0,k).slice(-4).map(isTiers?1:0);
    const alt=q[0]!==q[1]&&q[1]!==q[2]&&q[2]!==q[3];
    const curAlt=bits[0]!==bits[1]&&bits[1]!==bits[2]&&bits[2]!==bits[3];
    return alt===curAlt;
  });

  return out;
}

function combine(probs,weights){
  let num=0,den=0;
  for(const f of EXPERTS){
    if(probs[f] && probs[f].n>=1){
      const w=Math.max(.001,weights[f]||1);
      num+=w*logit(probs[f].p);den+=w;
    }
  }
  return den?sigmoid(num/den):.5;
}

/* Exponential-weights learning: each expert earns/loses weight according
   to its historical log loss. This is adaptive rather than fixed weighting. */
function learn(a){
  const c=a.slice().reverse();
  const w=Object.fromEntries(EXPERTS.map(x=>[x,1]));
  const preds=[];
  if(c.length<=MIN_TRAIN)return {weights:w,preds:[],brier:null,baseBrier:null,calibration:null};

  const eta=.10;
  for(let k=MIN_TRAIN;k<c.length;k++){
    const past=c.slice(0,k), ps=expertProbs(past);
    const p=combine(ps,w), y=isTiers(c[k])?1:0;
    preds.push({p,y});
    for(const f of EXPERTS){
      if(!ps[f] || ps[f].n<3)continue;
      const q=clamp(ps[f].p,.01,.99);
      const loss=-(y*Math.log(q)+(1-y)*Math.log(1-q));
      w[f]=clamp(w[f]*Math.exp(-eta*loss),.08,12);
    }
    const z=EXPERTS.reduce((s,f)=>s+w[f],0);
    for(const f of EXPERTS)w[f]/=z/EXPERTS.length;
  }
  if(!preds.length)return {weights:w,preds,brier:null,baseBrier:null,calibration:null};
  const brier=preds.reduce((s,q)=>s+(q.p-q.y)**2,0)/preds.length;
  let base=0;
  for(let k=MIN_TRAIN;k<c.length;k++){
    const past=c.slice(0,k), p=beta(past.filter(isTiers).length,past.length);
    base+=(p-(isTiers(c[k])?1:0))**2;
  }
  base/=preds.length;
  const bins=Array.from({length:5},()=>({n:0,p:0,y:0}));
  preds.forEach(q=>{const i=Math.min(4,Math.floor(q.p*5));bins[i].n++;bins[i].p+=q.p;bins[i].y+=q.y});
  let cal=0, ncal=0;
  bins.forEach(b=>{if(b.n){cal+=Math.abs(b.p/b.n-b.y/b.n)*b.n;ncal+=b.n}});
  return {weights:w,preds,brier,baseBrier:base,calibration:ncal?1-cal/ncal:null};
}

function currentModel(a){
  if(a.length<MIN_TRAIN)return {ready:false,p:.5,weights:null,probs:{}};
  const learned=learn(a), ps=expertProbs(a), p=combine(ps,learned.weights);
  const active=EXPERTS.filter(f=>ps[f]&&ps[f].n>=3).length;
  const edge=Math.abs(p-.5);
  const validation=learned.brier==null?0:clamp(1-learned.brier/.25,0,1);
  const separation=clamp(edge*2,0,1);
  const support=active/EXPERTS.length;
  let confidence="Baja";
  if(a.length>=80 && edge>=.12 && validation>=.15 && support>=.6)confidence="Alta";
  else if(a.length>=35 && edge>=.06 && validation>=.05 && support>=.4)confidence="Media";
  return {ready:true,p,weights:learned.weights,probs:ps,confidence,active,validation,edge,learned};
}

function render(){
  const a=state.history, m=currentModel(a);
  $("counter").textContent=`${a.length} spins`;$("spins").textContent=a.length;
  $("history").innerHTML=a.slice(0,50).map(n=>`<span class="result ${RED.has(n)?"red":n===0?"green":"black"}">${n}</span>`).join("")||"—";

  if(!m.ready){
    $("tiersProb").textContent="—";$("noTiersProb").textContent="—";$("mainSide").textContent="—";
    $("confidence").textContent=a.length?`Datos insuficientes (${a.length}/${MIN_TRAIN})`:"—";
    $("evidence").textContent="Insuficiente";$("signal").textContent="ESPERANDO DATOS";$("signal").className="signal";
    $("modelStatus").innerHTML=`<span class="warn">El modelo necesita ${MIN_TRAIN} spins para empezar a evaluar señales condicionales.</span>`;
    $("signals").innerHTML="—";$("validationCount").textContent="—";$("brier").textContent="—";$("baselineBrier").textContent="—";$("improvement").textContent="—";$("calibration").textContent="—";
    $("tiersBar").style.width="0";$("noTiersBar").style.width="0";return;
  }

  const p=m.p, np=1-p;
  $("tiersProb").textContent=pct(p);$("noTiersProb").textContent=pct(np);
  $("tiersBar").style.width=`${100*p}%`;$("noTiersBar").style.width=`${100*np}%`;
  $("mainSide").textContent=p>=.5?`Tiers · ${pct(p)}`:`No Tiers · ${pct(np)}`;
  $("confidence").textContent=m.confidence;
  $("evidence").textContent=`${m.active}/${EXPERTS.length} señales`;
  $("signal").textContent=m.confidence==="Alta"?"SEÑAL FUERTE":m.confidence==="Media"?"SEÑAL MODERADA":"SEÑAL DÉBIL";
  $("signal").className="signal "+(m.confidence==="Alta"?"high":m.confidence==="Media"?"medium":"");

  $("modelStatus").innerHTML=
    `<div><b>Estimación:</b> probabilidad modelada de Tiers en la próxima tirada.</div>
     <div><b>Motor:</b> ensemble adaptativo con pesos aprendidos mediante pérdida histórica walk-forward.</div>
     <div><b>Regla:</b> no existe un incremento mínimo ni un movimiento artificial desde 50%.</div>
     <div><b>Validación:</b> ${m.learned.preds.length} predicciones históricas evaluadas.</div>`;

  const names={recent:"Frecuencia reciente",transition:"Transiciones",interval:"Intervalos",streak:"Rachas",jump:"Magnitud de salto",avgJump3:"Media últimos 3 saltos",direction:"Dirección física",position:"Posición cilindro",sequence:"Secuencias",alternation:"Alternancia"};
  $("signals").innerHTML=EXPERTS.map(f=>{
    const q=m.probs[f], w=m.weights[f];
    if(!q)return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>—</span></div><small>Sin evidencia suficiente</small></div>`;
    const delta=q.p-.5;
    return `<div class="sig"><div class="sigtop"><b>${names[f]}</b><span>${pct(q.p)}</span></div><small>${q.n} casos · ${delta>=0?"+":""}${(delta*100).toFixed(1)} pp · peso <span class="weight">${w.toFixed(2)}×</span></small></div>`;
  }).join("");

  const l=m.learned;
  $("validationCount").textContent=`${l.preds.length} predicciones`;
  $("brier").textContent=l.brier==null?"—":l.brier.toFixed(4);
  $("baselineBrier").textContent=l.baseBrier==null?"—":l.baseBrier.toFixed(4);
  $("improvement").textContent=l.brier==null?"—":pct(clamp((l.baseBrier-l.brier)/Math.max(.0001,l.baseBrier),-1,1));
  $("calibration").textContent=l.calibration==null?"—":pct(clamp(l.calibration,0,1));
}

function add(n){
  const p=currentModel(state.history);
  state.predictions.unshift({spin:state.history.length+1,prediction:p.ready?p.p:null,actual:n,at:new Date().toISOString()});
  state.history.unshift(n);
  save();render();
}
function undo(){
  if(!state.history.length)return;
  state.history.shift();state.predictions.shift();save();render();
}
function clearAll(){
  if(confirm("¿Borrar todo el histórico?")){
    state={version:"4.0",history:[],predictions:[]};save();render();
  }
}

$("theme").onclick=()=>{
  document.body.classList.toggle("night");
  const night=document.body.classList.contains("night");
  localStorage.setItem("tiers-engine-theme",night?"night":"day");
  $("theme").textContent=night?"Modo día":"Modo noche";
};
$("undo").onclick=undo;$("clear").onclick=clearAll;

for(let n=0;n<=36;n++){
  const b=document.createElement("button");
  b.className=`num ${n===0?"green":RED.has(n)?"red":"black"}`;
  b.textContent=n;b.type="button";b.onclick=()=>add(n);$("numbers").appendChild(b);
}
load();
if(localStorage.getItem("tiers-engine-theme")==="night"){$("theme").click()}
render();
})();
