(() => {
"use strict";

const TIERS = new Set([27,13,36,11,30,8,23,10,5,24,16,33]);
const WHEEL = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];
const RED = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const STORE = "tiers-engine-v3-history";
let history = [];

const $ = id => document.getElementById(id);
const isTiers = n => TIERS.has(n);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const percent = x => `${(x*100).toFixed(1)}%`;

function readHistory(){
  try{
    const raw = localStorage.getItem(STORE);
    if(!raw) return [];
    const parsed = JSON.parse(raw);
    if(!Array.isArray(parsed)) return [];
    return parsed.filter(n => Number.isInteger(n) && n >= 0 && n <= 36);
  }catch(e){ console.error(e); return []; }
}
function writeHistory(){
  try{ localStorage.setItem(STORE, JSON.stringify(history)); }
  catch(e){ console.error("Unable to persist history",e); }
}
function rate(a){ return a.length ? a.filter(isTiers).length/a.length : .5; }
function streak(a,want){ let n=0; for(const x of a){if(isTiers(x)===want)n++;else break} return n; }
function currentInterval(a){
  for(let i=0;i<a.length;i++) if(isTiers(a[i])) return i;
  return a.length;
}
function averageInterval(a){
  const p=[]; for(let i=0;i<a.length;i++) if(isTiers(a[i])) p.push(i);
  if(p.length<2) return null;
  let s=0; for(let i=0;i<p.length-1;i++) s += p[i]-p[i+1]-1;
  return s/(p.length-1);
}
function distance(a,b){
  const ia=WHEEL.indexOf(a), ib=WHEEL.indexOf(b), d=Math.abs(ia-ib);
  return Math.min(d,WHEEL.length-d);
}
function dir(a,b){
  const ia=WHEEL.indexOf(a),ib=WHEEL.indexOf(b),n=WHEEL.length;
  const cw=(ib-ia+n)%n,ccw=(ia-ib+n)%n;
  if(cw===ccw)return 0; return cw<ccw?1:-1;
}
function avgLastJumps(a,k=3){
  if(a.length<2)return null;
  const ds=[]; for(let i=0;i<Math.min(k,a.length-1);i++)ds.push(distance(a[i],a[i+1]));
  return ds.reduce((x,y)=>x+y,0)/ds.length;
}

/* Feature vector uses only the supplied past array. */
function features(a){
  if(a.length<5)return null;
  const r=w=>rate(a.slice(0,Math.min(w,a.length)));
  const last=isTiers(a[0]);
  const prev=a.length>1?isTiers(a[1]):last;
  let tt=0,tn=0,nt=0,nn=0;
  for(let i=0;i<a.length-1;i++){
    const older=isTiers(a[i+1]), newer=isTiers(a[i]);
    if(older&&newer)tt++; else if(older&&!newer)nt++; else if(!older&&newer)tn++; else nn++;
  }
  const transition = last ? (tt+tn?tt/(tt+tn):.5) : (nt+nn?nt/(nt+nn):.5);
  const ai=averageInterval(a), ci=currentInterval(a);
  const jump=a.length>1?distance(a[0],a[1]):18, aj=avgLastJumps(a)??jump;
  const d=a.length>1?dir(a[1],a[0]):0;
  const pos=WHEEL.indexOf(a[0])/WHEEL.length-.5;
  return [
    2*r(5)-1,2*r(10)-1,2*r(20)-1,2*r(50)-1,2*r(100)-1,2*r(250)-1,
    2*r(a.length)-1,2*transition-1,last===prev?1:-1,
    clamp((streak(a,true)-streak(a,false))/8,-1,1),
    clamp((ci-(ai??ci))/10,-1,1),clamp((jump-18)/18,-1,1),
    clamp((aj-18)/18,-1,1),d,pos
  ];
}
const NF=15;

function sigmoid(z){return 1/(1+Math.exp(-clamp(z,-20,20)))}
function walkForward(a){
  if(a.length<16)return {preds:[],weights:new Array(NF+1).fill(0),brier:null,accuracy:null,bins:[]};
  let w=new Array(NF+1).fill(0),preds=[];
  const lr=.035,l2=.001;
  for(let t=a.length-1;t>=5;t--){
    const past=a.slice(t),x=features(past); if(!x)continue;
    const p=sigmoid(w[0]+x.reduce((s,v,i)=>s+w[i+1]*v,0));
    const y=isTiers(a[t-1])?1:0;
    preds.push({p,y});
    const e=p-y; w[0]=clamp(w[0]-lr*e,-4,4);
    for(let i=0;i<NF;i++)w[i+1]=clamp(w[i+1]-lr*(e*x[i]+l2*w[i+1]),-3,3);
  }
  if(!preds.length)return {preds,weights:w,brier:null,accuracy:null,bins:[]};
  const brier=preds.reduce((s,q)=>s+(q.p-q.y)**2,0)/preds.length;
  const accuracy=preds.filter(q=>(q.p>=.5?1:0)===q.y).length/preds.length;
  const bins=[0,1,2,3,4].map(()=>({n:0,p:0,y:0}));
  preds.forEach(q=>{const i=Math.min(4,Math.floor(q.p*5));bins[i].n++;bins[i].p+=q.p;bins[i].y+=q.y});
  return {preds,weights:w,brier,accuracy,bins};
}
function model(a){
  if(a.length<12)return {ready:false,p:.5,wf:null,x:null};
  const wf=walkForward(a),x=features(a),w=wf.weights;
  let p=sigmoid(w[0]+x.reduce((s,v,i)=>s+w[i+1]*v,0));
  const shrink=clamp((a.length-12)/88,0,1);
  p=0.5*(1-shrink)+p*shrink;
  p=clamp(.75*p+.25*rate(a),.02,.98);
  const edge=Math.abs(p-.5);
  const validation=wf.brier==null?0:clamp(1-wf.brier/.25,0,1);
  let conf="Low";
  if(a.length>=50 && edge>=.12 && validation>=.25)conf="High";
  else if(a.length>=25 && edge>=.06 && validation>=.10)conf="Medium";
  return {ready:true,p,conf,wf,x,w};
}
function contribution(v){if(Math.abs(v)<.005)return "Neutral";return `${v>0?"+":""}${(v*100).toFixed(1)} pp`}

function render(){
  const a=history,m=model(a),r=a.length?rate(a):null;
  $("total").textContent=a.length;$("histRate").textContent=r==null?"—":percent(r);
  $("tiersStreak").textContent=streak(a,true);$("noTiersStreak").textContent=streak(a,false);
  $("currentInterval").textContent=a.length?currentInterval(a):"—";
  const ai=averageInterval(a);$("averageInterval").textContent=ai==null?"—":ai.toFixed(1);
  $("brier").textContent=m.wf?.brier==null?"—":m.wf.brier.toFixed(4);
  $("accuracy").textContent=m.wf?.accuracy==null?"—":percent(m.wf.accuracy);

  $("modelState").textContent=m.ready?"MODEL ACTIVE":"DATA INSUFFICIENT";
  $("tiersPct").textContent=m.ready?percent(m.p):"—";
  $("noTiersPct").textContent=m.ready?percent(1-m.p):"—";
  $("mainPct").textContent=m.ready?percent(Math.max(m.p,1-m.p)):"—";
  $("mainSide").textContent=m.ready?(m.p>=.5?"Tiers":"No Tiers"):"Waiting for more results";
  $("confidence").textContent=m.ready?m.conf:"—";
  $("tiersBar").style.width=m.ready?`${m.p*100}%`:"50%";
  $("modelNote").textContent=m.ready?`Next-spin estimate · ${a.length} historical results · walk-forward validated.`:"Enter at least 12 results to activate the model.";

  const labels=["Frequency 5","Frequency 10","Frequency 20","Frequency 50","Frequency 100","Frequency 250","Complete history","Transition","Sequence","Streak","Interval","Last jump","Avg. last 3 jumps","Direction","Wheel position","Latest result"];
  let vals=a.length?[
    percent(rate(a.slice(0,5))),percent(rate(a.slice(0,10))),percent(rate(a.slice(0,20))),percent(rate(a.slice(0,50))),percent(rate(a.slice(0,100))),percent(rate(a.slice(0,250))),percent(rate(a)),
    m.ready?contribution(m.w[8]*m.x[7]):"—",m.ready?contribution(m.w[9]*m.x[8]):"—",m.ready?contribution(m.w[10]*m.x[9]):"—",
    m.ready?contribution(m.w[11]*m.x[10]):"—",m.ready?contribution(m.w[12]*m.x[11]):"—",m.ready?contribution(m.w[13]*m.x[12]):"—",m.ready?contribution(m.w[14]*m.x[13]):"—",m.ready?contribution(m.w[15]*m.x[14]):"—",
    `${a[0]} · ${isTiers(a[0])?"Tiers":"No Tiers"}`
  ]:labels.map(()=> "—");
  $("signals").innerHTML=labels.map((x,i)=>`<div class="signal"><span>${x}</span><strong>${vals[i]}</strong></div>`).join("");

  if(!m.wf?.brier){
    $("validation").innerHTML='<div class="validation-note">Walk-forward validation will appear after enough sequential observations are available.</div>';
  }else{
    const p0=rate(a),baseBrier=p0*(1-p0),imp=baseBrier?1-m.wf.brier/baseBrier:null;
    const cal=m.wf.bins.filter(b=>b.n).map(b=>`${percent(b.p/b.n)} pred → ${percent(b.y/b.n)} actual (${b.n})`).join(" · ");
    $("validation").innerHTML=`<div class="metric"><span>Predictions tested</span><strong>${m.wf.preds.length}</strong></div><div class="metric"><span>Brier score</span><strong>${m.wf.brier.toFixed(4)}</strong></div><div class="metric"><span>Accuracy</span><strong>${percent(m.wf.accuracy)}</strong></div><div class="metric"><span>Vs baseline</span><strong>${imp==null?"—":(imp>=0?"+":"")+percent(imp)}</strong></div><div class="validation-note">Calibration: ${cal||"not enough observations in bins"}</div>`;
  }

  $("historyCount").textContent=`${Math.min(a.length,50)} / 50`;
  if(!a.length){$("history").className="history empty";$("history").textContent="No results yet."}
  else{
    $("history").className="history";
    $("history").innerHTML=a.slice(0,50).map((n,i)=>{
      const d=i<a.length-1?distance(n,a[i+1]):null;
      const di=i<a.length-1?dir(a[i+1],n):0;
      return `<div class="history-row"><div class="number">${n}</div><div class="badge ${isTiers(n)?"tiers":"no-tiers"}">${isTiers(n)?"TIERS":"NO TIERS"}</div><div class="history-meta">${i===0?"LATEST":`${i} ago`}</div><div class="history-meta">${d==null?"—":d+" pockets"}</div><div class="history-meta">${di===1?"CW":di===-1?"CCW":"—"}</div></div>`;
    }).join("");
  }
}

function addResult(n){history.unshift(n);writeHistory();render()}
function start(){
  history=readHistory();
  document.querySelectorAll("[data-number]").forEach(btn=>btn.addEventListener("click",()=>addResult(Number(btn.dataset.number))));
  $("undoBtn").addEventListener("click",()=>{if(history.length){history.shift();writeHistory();render()}});
  $("clearBtn").addEventListener("click",()=>{if(history.length && window.confirm("Delete all stored results?")){history=[];writeHistory();render()}});
  $("themeBtn").addEventListener("click",()=>{document.body.classList.toggle("light");localStorage.setItem("tiers-engine-theme",document.body.classList.contains("light")?"light":"dark")});
  if(localStorage.getItem("tiers-engine-theme")==="light")document.body.classList.add("light");
  render();
}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",start,{once:true});else start();
})();