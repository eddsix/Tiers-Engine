const TIERS=[27,13,36,11,30,8,23,10,5,24,16,33];
const WHEEL=[0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26,0];
const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const HKEY="tiers-engine-history-v2";

let history=loadHistory();

function loadHistory(){
  try{
    const x=JSON.parse(localStorage.getItem(HKEY)||"[]");
    return Array.isArray(x)?x.map(Number).filter(n=>Number.isInteger(n)&&n>=0&&n<=36):[];
  }catch(e){return[]}
}
function save(){localStorage.setItem(HKEY,JSON.stringify(history))}
function isTiers(n){return TIERS.includes(n)}
function pct(x){return `${(x*100).toFixed(1)}%`}
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function rate(a){return a.length?a.filter(isTiers).length/a.length:.5}
function streak(a,want){let n=0;for(const x of a){if(isTiers(x)===want)n++;else break}return n}
function interval(a){
  for(let i=1;i<a.length;i++)if(isTiers(a[i]))return i-1;
  return a.length;
}
function avgInterval(a){
  const pos=[];for(let i=0;i<a.length;i++)if(isTiers(a[i]))pos.push(i);
  if(pos.length<2)return null;
  let s=0;for(let i=0;i<pos.length-1;i++)s+=pos[i]-pos[i+1]-1;
  return s/(pos.length-1);
}
function wheelDistance(a,b){
  const ia=WHEEL.indexOf(a),ib=WHEEL.indexOf(b),d=Math.abs(ia-ib);
  return Math.min(d,WHEEL.length-1-d);
}
function direction(a,b){
  const n=37,ia=WHEEL.indexOf(a),ib=WHEEL.indexOf(b);
  const cw=(ib-ia+n)%n,ccw=(ia-ib+n)%n;
  if(cw===ccw)return 0; return cw<ccw?1:-1;
}
function avgJump(a,k=3){
  if(a.length<2)return null;
  const ds=[];for(let i=0;i<Math.min(k,a.length-1);i++)ds.push(wheelDistance(a[i],a[i+1]));
  return ds.reduce((x,y)=>x+y,0)/ds.length;
}

/* Features are computed only from results before the prediction point. */
function features(a){
  const n=a.length;if(n<5)return null;
  const r=w=>rate(a.slice(0,Math.min(w,n)));
  const last=isTiers(a[0])?1:0;
  const prev=n>1?(isTiers(a[1])?1:0):.5;
  const transCount={tt:0,tn:0,nt:0,nn:0};
  for(let i=0;i<n-1;i++){
    const x=isTiers(a[i+1]),y=isTiers(a[i]);
    if(x&&y)transCount.tt++;else if(x&&!y)transCount.nt++;else if(!x&&y)transCount.tn++;else transCount.nn++;
  }
  const denT=transCount.tt+transCount.tn,denN=transCount.nt+transCount.nn;
  const afterLast=last?(denT?transCount.tt/denT:.5):(denN?transCount.nt/denN:.5);
  const seq=last===prev?1:-1;
  const noStreak=streak(a,false),tStreak=streak(a,true);
  const currentInterval=interval(a);
  const ai=avgInterval(a);
  const jump=a.length>1?wheelDistance(a[0],a[1]):18;
  const aj=avgJump(a,3)??jump;
  const dir=a.length>1?direction(a[1],a[0]):0;
  const lastPos=WHEEL.indexOf(a[0])/36-.5;
  return [
    2*r(5)-1,2*r(10)-1,2*r(20)-1,2*r(50)-1,2*r(100)-1,2*r(250)-1,
    2*r(n)-1,2*afterLast-1,seq,
    clamp((tStreak-noStreak)/8,-1,1),
    clamp((currentInterval-(ai??currentInterval))/10,-1,1),
    clamp((jump-18)/18,-1,1),
    clamp((aj-18)/18,-1,1),dir,lastPos
  ];
}
const NFEATURE=15;

function sigmoid(z){return 1/(1+Math.exp(-clamp(z,-20,20)))}
function dot(w,x){let s=w[0];for(let i=0;i<x.length;i++)s+=w[i+1]*x[i];return s}

/* Walk-forward learner. At every historical point, it predicts first, then learns
   from the actual next result. No future result is available to the prediction. */
function walkForward(a){
  if(a.length<16)return {predictions:[],brier:null,accuracy:null,weights:null,calibration:[]};
  let w=new Array(NFEATURE+1).fill(0);
  const preds=[];
  const lr=.035,l2=.001;
  for(let t=a.length-1;t>=5;t--){
    const past=a.slice(t);
    const x=features(past);
    if(!x)continue;
    const p=sigmoid(dot(w,x));
    const y=isTiers(a[t-1])?1:0;
    preds.push({p,y,x});
    const err=p-y;
    w[0]=clamp(w[0]-lr*err,-4,4);
    for(let j=0;j<NFEATURE;j++)w[j+1]=clamp(w[j+1]-lr*(err*x[j]+l2*w[j+1]),-3,3);
  }
  if(!preds.length)return {predictions:[],brier:null,accuracy:null,weights:w,calibration:[]};
  const brier=preds.reduce((s,q)=>s+(q.p-q.y)**2,0)/preds.length;
  const acc=preds.filter(q=>(q.p>=.5?1:0)===q.y).length/preds.length;
  const bins=[0,0,0,0,0].map(()=>({n:0,p:0,y:0}));
  preds.forEach(q=>{const i=Math.min(4,Math.floor(q.p*5));bins[i].n++;bins[i].p+=q.p;bins[i].y+=q.y});
  return {predictions:preds,brier,accuracy:acc,weights:w,calibration:bins};
}

function currentModel(a){
  if(a.length<12)return {ready:false,p:.5,confidence:"—",note:"At least 12 results are required before the adaptive model becomes active."};
  const wf=walkForward(a);
  const w=wf.weights||new Array(NFEATURE+1).fill(0);
  const x=features(a);
  let p=sigmoid(dot(w,x));
  /* With limited samples, shrink toward the empirical base rate. */
  const shrink=clamp((a.length-12)/88,0,1);
  p=.5*(1-shrink)+p*shrink;
  const base=rate(a);
  p=clamp(.75*p+.25*base,.02,.98);
  const edge=Math.abs(p-.5),quality=wf.brier==null?.5:clamp(1-wf.brier/.25,0,1);
  const sample=clamp(a.length/100,0,1);
  const score=edge*2*.55+quality*.25+sample*.20;
  const confidence=score>=.62?"High":score>=.40?"Medium":"Low";
  return {ready:true,p,confidence,wf,x,note:`Adaptive walk-forward model · ${a.length} results · no look-ahead.`};
}

function contributionRows(m){
  if(!m.ready)return [];
  const names=["5-result frequency","10-result frequency","20-result frequency","50-result frequency","100-result frequency","250-result frequency","Complete-history rate","Transition","Sequence","Streak","Interval","Last jump","Avg. last 3 jumps","Direction","Wheel position"];
  return names.map((name,i)=>({name,v:m.w[i+1]*m.x[i]}));
}
function formatContribution(v){
  if(Math.abs(v)<.005)return "Neutral";
  return `${v>0?"+":""}${(v*100).toFixed(1)} pp`;
}

function renderSignals(m){
  const a=history,items=[
    ["Frequency 5",a.length?pct(rate(a.slice(0,5))):"—"],
    ["Frequency 10",a.length?pct(rate(a.slice(0,10))):"—"],
    ["Frequency 20",a.length?pct(rate(a.slice(0,20))):"—"],
    ["Frequency 50",a.length?pct(rate(a.slice(0,50))):"—"],
    ["Frequency 100",a.length?pct(rate(a.slice(0,100))):"—"],
    ["Frequency 250",a.length?pct(rate(a.slice(0,250))):"—"],
    ["Complete history",a.length?pct(rate(a)):"—"],
    ["Transition",m.ready?formatContribution(m.w[8]*m.x[7]):"—"],
    ["Sequence",m.ready?formatContribution(m.w[9]*m.x[8]):"—"],
    ["Streak",m.ready?formatContribution(m.w[10]*m.x[9]):"—"],
    ["Interval",m.ready?formatContribution(m.w[11]*m.x[10]):"—"],
    ["Last jump",m.ready?formatContribution(m.w[12]*m.x[11]):"—"],
    ["Avg. last 3 jumps",m.ready?formatContribution(m.w[13]*m.x[12]):"—"],
    ["Direction",m.ready?formatContribution(m.w[14]*m.x[13]):"—"],
    ["Wheel position",m.ready?formatContribution(m.w[15]*m.x[14]):"—"],
    ["Current reference",a.length?`${a[0]} · ${isTiers(a[0])?"Tiers":"No Tiers"}`:"—"]
  ];
  $("signalsGrid").innerHTML=items.map(([k,v])=>`<div class="signal"><span>${k}</span><strong>${v}</strong></div>`).join("");
}
function renderValidation(m){
  const wf=m.wf;
  if(!wf||wf.brier==null){$("validation").innerHTML='<div class="validation-note">Validation becomes meaningful after enough sequential observations are available.</div>';return}
  const base=.25-(rate(history)-.5)**2;
  const improvement=base>0?1-wf.brier/base:null;
  const bins=wf.calibration.map((b,i)=>b.n?`<span>${i*20}–${i===4?100:(i+1)*20}%: ${pct(b.y/b.n)} actual (${b.n})</span>`:"").filter(Boolean).join(" · ");
  $("validation").innerHTML=`
    <div class="metric"><span>Predictions tested</span><strong>${wf.predictions.length}</strong></div>
    <div class="metric"><span>Brier score</span><strong>${wf.brier.toFixed(4)}</strong></div>
    <div class="metric"><span>Accuracy</span><strong>${pct(wf.accuracy)}</strong></div>
    <div class="metric"><span>Vs. baseline</span><strong>${improvement==null?"—":(improvement>=0?"+":"")+pct(improvement)}</strong></div>
    <div class="validation-note">Calibration: ${bins||"not enough observations in bins yet"}</div>`;
}
function render(){
  const a=history,m=currentModel(a),r=a.length?rate(a):null;
  $("total").textContent=a.length;$("histRate").textContent=r==null?"—":pct(r);
  $("tiersStreak").textContent=streak(a,true);$("noTiersStreak").textContent=streak(a,false);
  $("interval").textContent=a.length?interval(a):"—";
  const ai=avgInterval(a);$("avgInterval").textContent=ai==null?"—":ai.toFixed(1);
  $("brier").textContent=m.wf?.brier==null?"—":m.wf.brier.toFixed(4);
  $("accuracy").textContent=m.wf?.accuracy==null?"—":pct(m.wf.accuracy);
  $("status").textContent=m.ready?"MODEL ACTIVE":"DATA INSUFFICIENT";
  $("tiersPct").textContent=m.ready?pct(m.p):"—";$("noTiersPct").textContent=m.ready?pct(1-m.p):"—";
  $("mainPct").textContent=m.ready?pct(Math.max(m.p,1-m.p)):"—";
  $("mainSide").textContent=m.ready?(m.p>=.5?"Tiers":"No Tiers"):"Waiting for more results";
  $("confidence").textContent=m.confidence;$("tiersBar").style.width=m.ready?`${m.p*100}%`:"50%";$("modelNote").textContent=m.note||"";
  $("historyCount").textContent=`${Math.min(a.length,50)} / 50`;
  if(!a.length){$("history").className="history empty";$("history").textContent="No results yet."}
  else{
    $("history").className="history";
    $("history").innerHTML=a.slice(0,50).map((n,i)=>`
      <div class="history-row">
        <div class="number">${n}</div>
        <div class="badge ${isTiers(n)?"tiers":"no-tiers"}">${isTiers(n)?"TIERS":"NO TIERS"}</div>
        <div class="history-meta">${i===0?"LATEST":`${i} ago`}</div>
        <div class="history-meta">${i<a.length-1?`${wheelDistance(n,a[i+1])} pockets`:"—"}</div>
        <div class="history-meta">${i<a.length-1?direction(a[i+1],n)===1?"CW":direction(a[i+1],n)===-1?"CCW":"—":"—"}</div>
      </div>`).join("");
  }
  renderSignals(m);renderValidation(m);
}
function add(n){history.unshift(n);save();render()}
$("undoBtn").onclick=()=>{if(history.length){history.shift();save();render()}}
$("clearBtn").onclick=()=>{if(history.length&&confirm("Delete all stored results?")){history=[];save();render()}}
$("themeBtn").onclick=()=>document.body.classList.toggle("light");

for(let n=0;n<=36;n++){
  const b=document.createElement("button");
  b.textContent=n;
  b.classList.add(n===0?"roulette-green":RED.has(n)?"roulette-red":"roulette-black");
  b.setAttribute("aria-label",`Enter roulette result ${n}`);
  b.onclick=()=>add(n);
  $("keypad").appendChild(b);
}
render();
